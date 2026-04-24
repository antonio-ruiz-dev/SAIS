import os
import torch.distributed as dist
print("Lanzamiento distribuido funciona!")
os.environ["USE_LIBUV"] = "0"
print(' Distributed launch works')