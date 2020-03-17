import Predict
import BaseTransform
import json
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms

class_index = json.load(open('imagenet_class_index.json', 'r'))
ISLVRCPredictor = Predict.ILSVRCPredictor(class_index)

image_pathname = 'golden_retriever3.jpeg'
image = Image.open(image_pathname)

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

base_transform = BaseTransform.BaseTransform(resize, mean, std)
image_transform = base_transform(image)

image_transform = image_transform.unsqueeze_(0)

#image_transform = image_transform.numpy().transpose((0, 2, 3, 1))
#image_transform = np.clip(image_transform, 0, 1)

use_pretraining = True
net = models.vgg16(pretrained=use_pretraining)
net.eval()

result = net(image_transform)
class_name = ISLVRCPredictor.predict_max(result)

print('classification result: ', class_name)