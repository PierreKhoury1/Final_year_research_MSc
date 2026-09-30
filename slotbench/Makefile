# slotbench build. `make` compiles everything (no GPU needed to compile); `make test` builds and runs
# the host-only tests (no CUDA needed); `make pytest` runs the analysis tests.
# Override the target GPUs with e.g. `make SM=86` (RTX 3060) or `make SM="80 90"`.

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
PYTHON    ?= python3
SM        ?= 80 86 89 90

GENCODE   := $(foreach s,$(SM),-gencode arch=compute_$(s),code=sm_$(s)) \
             -gencode arch=compute_$(lastword $(SM)),code=compute_$(lastword $(SM))
INC       := -Icommon -Iphy
CXXFLAGS  ?= -O2 -g
CXXFLAGS  += -std=c++17 -Wall -Wextra $(INC)
NVFLAGS   := -O2 -g -lineinfo -std=c++17 $(GENCODE) $(INC) -Xcompiler -Wall
CUDA_LIBS := -lcublas -lcufft -lpthread -ldl

COMMON_H  := $(wildcard common/*.h)
PHY_H     := $(wildcard phy/*.h phy/*.cuh)

BINS  := bin/slot_driver bin/adversary bin/clockcal bin/phc_offset
TESTS := bin/test_ldpc bin/test_phy_ops bin/test_ring bin/test_clock_fit

.PHONY: all gpu host test pytest tables clean
all: $(BINS) $(TESTS)
gpu: bin/slot_driver bin/adversary bin/clockcal
host: bin/phc_offset $(TESTS)

build bin:
	mkdir -p $@

# ---- CUDA objects ----
build/%.o: phy/%.cu $(COMMON_H) $(PHY_H) | build
	$(NVCC) $(NVFLAGS) -c -o $@ $<
build/ldpc_code.o: phy/ldpc_code.cpp phy/ldpc_code.h phy/ldpc_tables.h phy/phy_ops.h $(COMMON_H) | build
	$(CXX) $(CXXFLAGS) -c -o $@ $<
build/slot_driver.o: driver/slot_driver.cu $(COMMON_H) $(PHY_H) | build
	$(NVCC) $(NVFLAGS) -c -o $@ $<

bin/slot_driver: build/slot_driver.o build/slot_pipeline.o build/phy_kernels.o build/ldpc_code.o | bin
	$(NVCC) $(NVFLAGS) -o $@ $^ $(CUDA_LIBS)

bin/adversary: adversary/adversary.cu $(COMMON_H) | bin
	$(NVCC) $(NVFLAGS) -o $@ $< $(CUDA_LIBS)

bin/clockcal: clock/clockcal.cu $(COMMON_H) | bin
	$(NVCC) $(NVFLAGS) -o $@ $< $(CUDA_LIBS)

# ---- host-only ----
bin/phc_offset: clock/phc_offset.c | bin
	$(CC) -O2 -g -std=gnu11 -Wall -Wextra -o $@ $<

bin/test_ldpc: tests/test_ldpc.cpp build/ldpc_code.o | bin
	$(CXX) $(CXXFLAGS) -o $@ $^
bin/test_phy_ops: tests/test_phy_ops.cpp phy/phy_ops.h $(COMMON_H) | bin
	$(CXX) $(CXXFLAGS) -o $@ $<
bin/test_ring: tests/test_ring.cpp $(COMMON_H) | bin
	$(CXX) $(CXXFLAGS) -o $@ $< -lpthread
bin/test_clock_fit: tests/test_clock_fit.cpp $(COMMON_H) | bin
	$(CXX) $(CXXFLAGS) -o $@ $<

test: $(TESTS)
	@set -e; for t in $(TESTS); do echo "== $$t"; ./$$t; done

pytest:
	$(PYTHON) -m pytest -q analysis/tests

tables:
	$(PYTHON) phy/gen_tables.py phy/tables phy/ldpc_tables.h

clean:
	rm -rf build bin
