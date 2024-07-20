import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(Net, self).__init__()
        # need to align in channels with image - 1 channel if grayscale, 3 for rgb... acutally in this case its 30 because the 3d scans are 512x512x30 
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, padding='same')  # set the size of the convolution to 5x5, and have 12 of them
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(256, 100) # changed from 50 to 100 since 75 classes... dont know if this actually matteres
        self.fc2 = nn.Linear(100, n_classes)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
    
    def forward(self, x):
        x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = F.relu(self.bn2(F.max_pool2d(self.conv2(x), 2)))
        x = F.relu(self.bn3(F.max_pool2d(self.conv3(x), 2)))
        x = F.relu(self.bn4(F.max_pool2d(self.conv4(x), 2)))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# # trying slightly larger net to see if it makes any difference
# class Net(nn.Module):
#     def __init__(self, in_channels, n_classes):
#         super(Net, self).__init__()
#         # need to align in channels with image - 1 channel if grayscale, 3 for rgb... acutally in this case its 30 because the 3d scans are 512x512x30 
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, padding='same')  # set the size of the convolution to 5x5, and have 12 of them
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding='same')
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
#         self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
#         self.fc1 = nn.Linear(512, 100) # changed from 50 to 100 since 75 classes... dont know if this actually matteres
#         self.fc2 = nn.Linear(100, n_classes)
        
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
#         self.bn1 = nn.BatchNorm2d(32)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm2d(512)
    
#     def forward(self, x):
#         x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
#         x = F.relu(self.bn2(F.max_pool2d(self.conv2(x), 2)))
#         x = F.relu(self.bn3(F.max_pool2d(self.conv3(x), 2)))
#         x = F.relu(self.bn4(F.max_pool2d(self.conv4(x), 2)))
#         x = F.relu(self.bn5(F.max_pool2d(self.conv5(x), 2)))
#         x = self.global_avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

