#!/bin/bash
# on-box: build, dump SASS, run, pack
set -x
mkdir -p /root/res && cd /root && tar xzf /root/tickbound.tgz
{
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv
nvcc --version | tail -2
nvcc -O3 -arch=sm_80 -lineinfo -o reorder_proof reorder_proof.cu 2>&1
cuobjdump -sass reorder_proof > res/proof_sm80.sass
nvcc -O3 -arch=sm_80 -Xptxas -v -o /dev/null reorder_proof.cu 2>&1 | grep -E "registers|Compiling"
./reorder_proof | tee res/proof.csv
} > res/onbox.log 2>&1
cp reorder_proof.cu res/
tar czf /root/res.tgz res && touch /root/DONE
