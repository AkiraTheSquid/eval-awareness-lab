"""Raw matmul throughput by thread count: is the CPU path healthy?"""
import time

import torch as t

for n in (1, 4, 8, 12, 16):
    t.set_num_threads(n)
    a = t.randn(2048, 2048)
    a @ a
    s = time.time()
    for _ in range(10):
        a @ a
    dt = (time.time() - s) / 10
    print(n, "threads", round(2 * 2048 ** 3 / dt / 1e9), "GFLOPS")
