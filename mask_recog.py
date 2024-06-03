import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
        # print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from matplotlib import pyplot as plt
import cv2
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import tqdm
import random
import numpy as np
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset
from openpyxl import Workbook
from PIL import Image
import time
import os
import torch



class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None


    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out







savepath='./picture' 
if not os.path.exists(savepath):
    os.mkdir(savepath)

def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        # plt.tight_layout()
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
        print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))

class ft_net(nn.Module):

    def __init__(self):
        super(ft_net, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        self.model = model_ft


    def forward(self, x):
        if False: # draw features or not
            x = self.model.conv1(x)
            draw_features(1,1,x.cpu().numpy(),"{}/conv1.png".format(savepath))

            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
          
            x = self.model.layer1(x)
            draw_features(1, 1, x.cpu().numpy(), "{}/layer1.png".format(savepath))

            x = self.model.layer2(x)
            draw_features(1, 1, x.cpu().numpy(), "{}/layer2.png".format(savepath))

            x = self.model.layer3(x)
            draw_features(1, 1, x.cpu().numpy(), "{}/layer3.png".format(savepath))

            x = self.model.layer4(x)
            draw_features(1, 1, x.cpu().numpy()[:, 0:1024, :, :], "{}/layer4.png".format(savepath))
         
            x = self.model.avgpool(x)
            plt.clf()
            plt.close()

            x = x.view(x.size(0), -1)
            x = self.model.fc(x)
            # plt.plot(np.linspace(1, 1000, 1000), x.cpu().numpy()[0, :])
            # plt.savefig("{}/f10_fc.png".format(savepath))
            plt.clf()
            plt.close()
        else :
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x=se(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.model.fc(x)

        return x








class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y








class CustomDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.classes = sorted(os.listdir(dataset_path))

    def __len__(self):
        return sum(len(files) for _, _, files in os.walk(self.dataset_path))

    def __getitem__(self, idx):
        class_idx = 0
        while idx >= len(os.listdir(os.path.join(self.dataset_path, self.classes[class_idx]))):
            idx -= len(os.listdir(os.path.join(self.dataset_path, self.classes[class_idx])))
            class_idx += 1
        class_path = os.path.join(self.dataset_path, self.classes[class_idx])
        file_name = os.listdir(class_path)[idx]


        #face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
        image = cv2.imread(os.path.join(class_path, file_name))
        image = cv2.resize(image,(128,128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

        #faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    


        image_pil = Image.fromarray(image)  # 转换为PIL图像

        if self.transform:
            image_pil = self.transform(image_pil)

        return image_pil, class_idx



if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    epochs = 20
    lr = 0.005
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    dataset_path = 'your_dataset_path'
    print(dataset_path)

  


    full_dataset_opencv = CustomDataset(dataset_path, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor()
    ]))







    #full_dataset = ImageFolder(dataset_path, transform=transforms.Compose([transforms.Resize((128, 128)), transforms.RandomHorizontalFlip(p=0.2),transforms.ToTensor()]))
    len(full_dataset_opencv)
    # %20 of dataset is test data
    train_dataset, val_dataset = random_split(full_dataset_opencv, [1942, 600])

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    net=ft_net()


    se = CoordAtt(inp=128,oup=128)



    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=lr)

    train_loss = []
    train_acc = []
    val_acc = []
    val_loss = []

    for epoch in range(epochs):
        loss_sum = 0.0
        correct = 0
        total = 0
        loss_sum2 = 0.0
        data_iter = tqdm.tqdm(
            enumerate(train_dl),
            desc=f"Train EP_{epoch+1}",
            total=len(train_dl),
        )
        for i, (images, labels) in data_iter:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            print(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_l = loss_sum / len(train_dl)
        train_a = correct / total

        correct = 0
        total = 0
        data_iter = tqdm.tqdm(
            val_dl,
            desc=f"Val EP_{epoch+1}",
            total=len(val_dl),
        )
        for images, labels in data_iter:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss2 = criterion(outputs, labels)
            loss_sum2 += loss2.item()
        val_a = correct / total
        val_l=loss_sum2/len(val_dl)

        print(
            f"Epoch: {epoch + 1}, Train Average Loss: {train_l:.6f}, val_loss:{val_l:.6f},Train Acc: {train_a*100:.2f}%, Val Acc: {val_a*100:.2f}%"
        )
        train_loss.append(train_l)
        train_acc.append(train_a)
        val_acc.append(val_a)
        val_loss.append(val_l)



    wb = Workbook()


    ws = wb.active
    ws.title = "Arrays Data" 

    arrays = [train_loss, train_acc, val_acc, val_loss]


    for i, array in enumerate(arrays, start=1):
        for j, value in enumerate(array, start=1):
            ws.cell(row=j, column=i, value=value)


    wb.save("save_excel.xlsx")

    num_ftrs = net.model.fc.in_features
    net.model.fc = nn.Linear(num_ftrs, 3) 


    torch.save(net, 'savepath.pth')