import torch,torchvision
from PIL import Image


image_path="E:/p_nyu2/home_office_0001_out/rgb/1.png"
image=Image.open(image_path)

image_tensor=torchvision.transforms.ToTensor()(image)

print(image_tensor.shape)