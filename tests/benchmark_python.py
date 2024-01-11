import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
from PIL import Image
from torchvision import transforms
import time
import os

current_path = os.getcwd()
image_path = os.path.join(current_path, "./src/images/pelican.jpeg")
input_image = Image.open(image_path)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) 
num_iters = 100
with torch.no_grad():
    start = time.time()
    for _ in range(num_iters):
        output = model(input_batch)
    end = time.time()
    print(((end - start) * 1000) / num_iters)
