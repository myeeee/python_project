from torchvision import models, transforms

class image_transform:
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train':transforms.Compose([transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)]),
            'val':transforms.Compose([transforms.Resize(resize),
                                        transforms.CenterCrop(resize),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
        }
    
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform_obj = image_transform(resize, mean, std)

