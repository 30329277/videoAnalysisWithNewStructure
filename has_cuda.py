# import torch

# print("CUDA Available:", torch.cuda.is_available())

import torch

print("CUDA Available:", torch.cuda.is_available())
print("cuDNN Enabled:", torch.backends.cudnn.enabled)
if torch.cuda.is_available():
    print("Current CUDA Device Index:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print(torch.__version__)
    print(torch.version.cuda)

