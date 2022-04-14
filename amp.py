# %%
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader 
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
torch.backends.cudnn.benchmark = True

# %%
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, device='cpu'):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.device = device
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        # targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data, 1)# for mldg da
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)#for zzd
        targets = targets.to(self.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-Variable(targets) * log_probs).mean(0).sum()
        return loss

# %%
class FoodDataset(Dataset):
    def __init__(self, file, transform=None, mode='train'):
        self.transforms = transform
        self.mode = mode
        with open(file, 'r') as f:
            self.image_list = f.readlines()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        label = None
        if self.mode == 'train':
            image, label = self.image_list[index].split('\n')[0].split('\t')
            label = int(label)
        else:
            image = self.image_list[index].split('\n')[0]
        image = Image.open(image).convert('RGB')
        image = self.transforms(image)
        if self.mode == 'train':
            return image, label
        else:
            return image

# %%
transforms_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Pad(10, 10),
                transforms.RandomRotation(45),
                transforms.RandomCrop((224, 224)),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

transforms_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

# %%
def evaluate(prediction, ground_truth):
    num_correct = (np.array(prediction) == np.array(ground_truth)).sum()
    return num_correct / len(prediction)

# %%
train_ds = FoodDataset('/media/ntu/volume2/home/s121md302_07/food/data/train.txt', transform=transforms_train)
val_ds = FoodDataset('/media/ntu/volume2/home/s121md302_07/food/data/val.txt', transform=transforms_test)
test_ds = FoodDataset('/media/ntu/volume2/home/s121md302_07/food/data/test.txt', transform=transforms_test)

batch_size = 128
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=8)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=8)

# %%
num_classes = 5
train_model = models.resnet34(pretrained=True)
train_model.fc = nn.Linear(512, num_classes)

model_str = 'resnet34'
output_dir = 'checkpoint_' + model_str
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ce_loss = CrossEntropyLabelSmooth(num_classes = num_classes, device = device)
optimizer = torch.optim.Adam(train_model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
scaler = torch.cuda.amp.GradScaler()

# %%
# train_model.load_state_dict(torch.load('checkpoint_resnet34/resnet34_50.pth'))

# %%
import time
train_model.eval()
temp = torch.rand([1, 3, 224, 224])
with torch.no_grad():
    start = time.time()
    out = train_model(temp)
    end = time.time()
    print('Time taken for forward pass without AMP: {}'.format(end - start))

with torch.no_grad():
    with torch.cuda.amp.autocast():
        start = time.time()
        out = train_model(temp)
        end = time.time()
        print('Time taken for forward pass with AMP: {}'.format(end - start))

# %%
import torch, time, gc

start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

# %%
scaler = torch.cuda.amp.GradScaler(enabled = False)
train_model.train()
train_model.to(device)
epoch = 10
start_timer()
for ep in range(epoch):
    for img, label in tqdm(train_dl):
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = False):
            output= train_model(img)
            assert output.dtype is torch.float32
            loss = ce_loss(output, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    end = time.time()
end_timer_and_print("Default precision:")

# %%
# print('Time taken for 1 epoch without AMP: {}'.format(sum(time_list)/len(time_list)))

# %%
scaler = torch.cuda.amp.GradScaler()
train_model.train()
train_model.to(device)
epoch = 10
start_timer()
for ep in range(epoch):
    for img, label in tqdm(train_dl):
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output= train_model(img)
            assert output.dtype is torch.float16
            loss = ce_loss(output, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    end = time.time()
end_timer_and_print("Default precision:")


 

# %%
# print('Time taken for 1 epoch with AMP: {}'.format(sum(time_list)/len(time_list)))
torch.cuda.empty_cache()

# %%



