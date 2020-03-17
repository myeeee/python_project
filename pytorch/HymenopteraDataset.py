import torch.utils.data as data
from PIL import Image
import image_transform
import dataname_list
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch

class HymenopteraDataset(data.Dataset):
    
    def __init__(self, filelist, transform=None, phase='train'):
        self.filelist = filelist
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):

        file_path = self.filelist[index]
        img = Image.open(file_path)

        img_transform = self.transform(img, self.phase)

        if self.phase == 'train':
            label = self.filelist[index][23:27]
        elif self.phase == 'val':
            label = self.filelist[index][21:25]

        if label == 'ants':
            label = 0
        elif label == 'bees':
            label = 1

        return img_transform, label


dataset_train = HymenopteraDataset(dataname_list.train_list, image_transform.transform_obj, phase='train')
dataset_val = HymenopteraDataset(dataname_list.val_list, image_transform.transform_obj, phase='val')

batch_size = 32

train_dataloader = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

# 辞書型変数にまとめる
dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader}

#batch_iterator = iter(dataloaders_dict['train'])
#inputs, labels = next(batch_iterator)
#print(inputs.size())
#print(labels)

use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)

net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

# 訓練モードに設定
net.train()

print('ネットワーク設定完了：学習済みの重みをロードし、訓練モードに設定しました')

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# 最適化手法を設定
params_update = []
update_params_names = ['classifier.6.weight','classifier.6.bias']

for name, param in net.named_parameters():
    if name in update_params_names:
        param.requires_grad = True
        params_update.append(param)
        print(name)
    else:
        param.requires_grad = False

print("------------------")
print(params_update)

# 最適化手法の設定
optimizer = optim.SGD(params=params_update, lr=0.001, momentum=0.9)

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch: {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            
            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

num_epochs=2
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
