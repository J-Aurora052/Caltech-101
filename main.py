import torch
# 强制指定GPU0并验证
torch.cuda.set_device(1)
print(f"当前GPU: {torch.cuda.current_device()}")
print(f"可用GPU数: {torch.cuda.device_count()}")
print(f"GPU0内存状态: {torch.cuda.memory_allocated(1)/1024**2:.1f}MB/{torch.cuda.max_memory_allocated(0)/1024**2:.1f}MB")