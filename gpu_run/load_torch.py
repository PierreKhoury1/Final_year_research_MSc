"""Realistic 'AI job' in a separate process: back-to-back fp16 matrix multiplies, like transformer inference."""
import sys
import time

import torch

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 60
x = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
w = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
end = time.time() + secs
n = 0
while time.time() < end:
    y = x @ w
    n += 1
    if n % 20 == 0:
        torch.cuda.synchronize()
torch.cuda.synchronize()
print(f"load_torch: {n} matmuls", file=sys.stderr)
