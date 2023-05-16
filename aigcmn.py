import torch
flag = torch.cuda.is_available()
print(flag)

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))


class AiGcMn:
    def __init__(self, ...):
        # 加载模型、设置参数等
        ...
    
    def misc(self, ...):
        # 其他处理函数
        ...
        
    def generate(self, target : torch.Tensor) -> torch.Tensor:
        # 根据target生成图像
        ...
        return imgs