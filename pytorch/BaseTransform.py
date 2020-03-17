from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class BaseTransform():
    def __init__(self, resize, mean, std):
        self.transform_base = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def __call__(self, img):
        return self.transform_base(img)

if __name__ == '__main__':
    img_path = 'golden_retriever.jpeg'
    image = Image.open(img_path)

    plt.imshow(image)
    plt.show()

    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transBase = BaseTransform(resize, mean, std)

    # リサイズ & 標準化
    image_transform = transBase(image)

    image_transform = image_transform.numpy().transpose((1, 2, 0))
    image_transform = np.clip(image_transform, 0, 1)

    plt.imshow(image_transform)
    plt.show()