import torch
from lib.models.refine_module2 import *


pretrained_path = "E:/pretrained/resnet18-5c106cde.pth"
# residual=Residual(pretrained_resnet18_path=pretrained_path)
# resample=Resample(pretrained_resnet18_path=pretrained_path)
refine_model = Refine_module2(pretrained_resnet18_path=pretrained_path)

img_tensor=torch.rand(2,3,385,385)
depth_tensor=torch.rand(2,1,385,385)
input=torch.rand(2,4,480,640)

# residual(input)
# resample(input)
