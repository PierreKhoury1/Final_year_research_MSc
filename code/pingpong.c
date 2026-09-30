// CPU-GPU clock ping-pong over fine-grained SVM.
// Host: tsc0 -> write seq -> spin until GPU acks -> tsc1. GPU: see seq -> read REALTIME -> ack.
// The GPU clock read is bracketed by [tsc0, tsc1] with no kernel launch in the loop.
#include <windows.h>
#include <intrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int32_t cl_int; typedef uint32_t cl_uint; typedef uint64_t cl_ulong; typedef cl_ulong cl_bitfield;
typedef void *cl_platform_id, *cl_device_id, *cl_context, *cl_command_queue, *cl_program, *cl_kernel, *cl_mem, *cl_event;
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_SVM_FINE_GRAIN_BUFFER (1 << 10)
#define CL_PROGRAM_BUILD_LOG 0x1183

#define FN(ret, name, args) typedef ret (__stdcall *name##_t) args; static name##_t name;
FN(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id *, cl_uint *))
FN(cl_int, clGetDeviceIDs, (cl_platform_id, cl_bitfield, cl_uint, cl_device_id *, cl_uint *))
FN(cl_context, clCreateContext, (const intptr_t *, cl_uint, const cl_device_id *, void *, void *, cl_int *))
FN(cl_command_queue, clCreateCommandQueueWithProperties, (cl_context, cl_device_id, const cl_ulong *, cl_int *))
FN(cl_program, clCreateProgramWithSource, (cl_context, cl_uint, const char **, const size_t *, cl_int *))
FN(cl_int, clBuildProgram, (cl_program, cl_uint, const cl_device_id *, const char *, void *, void *))
FN(cl_int, clGetProgramBuildInfo, (cl_program, cl_device_id, cl_uint, size_t, void *, size_t *))
FN(cl_kernel, clCreateKernel, (cl_program, const char *, cl_int *))
FN(void *, clSVMAlloc, (cl_context, cl_bitfield, size_t, cl_uint))
FN(cl_int, clSetKernelArgSVMPointer, (cl_kernel, cl_uint, const void *))
FN(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void *))
FN(cl_int, clEnqueueNDRangeKernel, (cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *))
FN(cl_int, clFlush, (cl_command_queue))
FN(cl_int, clFinish, (cl_command_queue))
FN(cl_mem, clCreateBuffer, (cl_context, cl_bitfield, size_t, void *, cl_int *))

static const char *SRC =
"#define STOP 0xFFFFFFFEu\n"
"__kernel void pp(__global volatile uint *ctl, __global ulong *gt, uint n, uint base){\n"
"  for(uint i=0;i<n;i++){\n"
"    uint want=base+i+1; uint c; ulong spins=0;\n"
"    while((c=ctl[0])!=want){ if(c==STOP||++spins>400000000ul) return; }\n"
"    ulong t=__builtin_amdgcn_s_sendmsg_rtnl(131);\n"
"    gt[i]=t; mem_fence(CLK_GLOBAL_MEM_FENCE);\n"
"    ctl[1]=want;\n"
"  }\n"
"}\n"
"__kernel void burn(__global float *a){ float v=a[get_global_id(0)];\n"
"  for(int i=0;i<4000;i++) v=v*1.0000001f+0.5f; a[get_global_id(0)]=v; }\n";

#define STOP 0xFFFFFFFEu
#define BATCH 400

typedef struct { uint32_t phase; uint32_t ok; uint64_t tsc0, tsc1, gpu; } rec_t;

static cl_context ctx; static cl_device_id dev;
static cl_command_queue q, q2; static cl_kernel kpp, kburn;
static volatile uint32_t *ctl; static uint64_t *gt;
static volatile LONG stop_load;

static inline uint64_t tsc(void) { _mm_lfence(); uint64_t t = __rdtsc(); _mm_lfence(); return t; }

static DWORD WINAPI cpu_spin(LPVOID p) { (void)p; volatile uint64_t x = 0; while (!stop_load) x++; return 0; }
static DWORD WINAPI gpu_burn(LPVOID p) {
    (void)p; size_t g = 1 << 20;
    while (!stop_load) { clEnqueueNDRangeKernel(q2, kburn, 1, NULL, &g, NULL, 0, NULL, NULL); clFinish(q2); }
    return 0;
}

static uint32_t seq;
// One phase: repeated batches of BATCH ping-pongs; sleep_us > 0 idles the host between samples.
static size_t run_phase(uint32_t ph, double secs, int sleep_us, rec_t *out, size_t cap) {
    size_t n = 0; LARGE_INTEGER f, a, b; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&a);
    for (;;) {
        QueryPerformanceCounter(&b);
        if ((double)(b.QuadPart - a.QuadPart) / f.QuadPart > secs || n + BATCH > cap) break;
        uint32_t base = seq, nb = BATCH; size_t g = 1;
        ctl[0] = base; ctl[1] = base;
        clSetKernelArgSVMPointer(kpp, 0, (void *)ctl); clSetKernelArgSVMPointer(kpp, 1, gt);
        clSetKernelArg(kpp, 2, 4, &nb); clSetKernelArg(kpp, 3, 4, &base);
        clEnqueueNDRangeKernel(q, kpp, 1, NULL, &g, NULL, 0, NULL, NULL); clFlush(q);
        size_t first = n; int aborted = 0;
        for (uint32_t i = 0; i < nb; i++) {
            uint32_t want = base + i + 1;
            if (sleep_us > 0) Sleep(sleep_us / 1000 ? sleep_us / 1000 : 1);
            uint64_t t0 = tsc(); ctl[0] = want;
            uint64_t lim = t0 + 3000000000ull; int ok = 1;  // ~1 s timeout
            while (ctl[1] != want) { if (__rdtsc() > lim) { ok = 0; break; } }
            uint64_t t1 = tsc();
            out[n].phase = ph; out[n].ok = ok; out[n].tsc0 = t0; out[n].tsc1 = t1; out[n].gpu = 0; n++;
            if (!ok) { aborted = 1; break; }
        }
        ctl[0] = STOP; clFinish(q);
        for (size_t k = first; k < n; k++) if (out[k].ok) out[k].gpu = gt[k - first];
        seq = base + nb + 1;
        if (aborted) fprintf(stderr, "phase %u: timeout, batch aborted\n", ph);
    }
    return n;
}

int main(int argc, char **argv) {
    double secs = argc > 1 ? atof(argv[1]) : 60;
    HMODULE h = LoadLibraryA("OpenCL.dll"); if (!h) { puts("no OpenCL.dll"); return 1; }
#define LOAD(name) name = (name##_t)GetProcAddress(h, #name); if (!name) { puts("missing " #name); return 1; }
    LOAD(clGetPlatformIDs) LOAD(clGetDeviceIDs) LOAD(clCreateContext) LOAD(clCreateCommandQueueWithProperties)
    LOAD(clCreateProgramWithSource) LOAD(clBuildProgram) LOAD(clGetProgramBuildInfo) LOAD(clCreateKernel)
    LOAD(clSVMAlloc) LOAD(clSetKernelArgSVMPointer) LOAD(clSetKernelArg) LOAD(clEnqueueNDRangeKernel)
    LOAD(clFlush) LOAD(clFinish) LOAD(clCreateBuffer)
    cl_platform_id plat; cl_int e;
    clGetPlatformIDs(1, &plat, NULL); clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e);
    q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &e);
    q2 = clCreateCommandQueueWithProperties(ctx, dev, NULL, &e);
    cl_program p = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &e);
    if (clBuildProgram(p, 1, &dev, "-cl-std=CL2.0", NULL, NULL)) {
        static char log[65536]; clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL); puts(log); return 1;
    }
    kpp = clCreateKernel(p, "pp", &e); kburn = clCreateKernel(p, "burn", &e);
    ctl = clSVMAlloc(ctx, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, 256, 0);
    gt = clSVMAlloc(ctx, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, BATCH * 8, 0);
    if (!ctl || !gt) { puts("SVM alloc failed"); return 1; }
    cl_mem big = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4 << 20, NULL, &e); clSetKernelArg(kburn, 0, sizeof big, &big);
    timeBeginPeriod(1);
    SYSTEM_INFO si; GetSystemInfo(&si);
    DWORD hcore = argc > 2 ? (DWORD)atoi(argv[2]) : 0;  // host thread core
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << hcore);
    if (argc > 3 && atoi(argv[3])) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    const char *outf = argc > 4 ? argv[4] : "pp.bin";

    size_t cap = 20000000; rec_t *r = malloc(cap * sizeof *r); size_t n = 0;
    LARGE_INTEGER qf, qa, qb; uint64_t ta, tb; QueryPerformanceFrequency(&qf);
    QueryPerformanceCounter(&qa); ta = tsc();
    const char *names[] = {"tight", "sleepy", "cpu_load", "gpu_load", "tight2"};
    HANDLE th[64]; int nth = 0;
    for (uint32_t ph = 0; ph < 5; ph++) {
        stop_load = 0; nth = 0;
        if (ph == 2) for (DWORD c = 0; c < si.dwNumberOfProcessors && nth < 64; c++) { if (c == hcore) continue;
            th[nth] = CreateThread(NULL, 0, cpu_spin, NULL, CREATE_SUSPENDED, NULL);
            SetThreadAffinityMask(th[nth], (DWORD_PTR)1 << c); ResumeThread(th[nth]); nth++; }
        if (ph == 3) th[nth++] = CreateThread(NULL, 0, gpu_burn, NULL, 0, NULL);
        size_t m = run_phase(ph, ph == 1 ? secs / 2 : secs, ph == 1 ? 1000 : 0, r + n, cap - n);
        stop_load = 1; WaitForMultipleObjects(nth, th, TRUE, INFINITE);
        size_t ok = 0; for (size_t k = n; k < n + m; k++) ok += r[k].ok;
        printf("%-9s %zu samples, %zu ok\n", names[ph], m, ok); fflush(stdout);
        n += m;
    }
    QueryPerformanceCounter(&qb); tb = tsc();
    double tsc_hz = (double)(tb - ta) / ((double)(qb.QuadPart - qa.QuadPart) / qf.QuadPart);
    printf("TSC %.3f MHz (vs QPC)\n", tsc_hz / 1e6);
    FILE *fo = fopen(outf, "wb"); fwrite(&tsc_hz, 8, 1, fo); fwrite(r, sizeof *r, n, fo); fclose(fo);
    return 0;
}
