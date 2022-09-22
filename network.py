# 50000 training images and 10000 test images with 3x32x32 shape
# 10 classes: {'airplane':0, 'automibile':1, 'bird':2, 'cat':3, 'deer':4, 
# 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(12)
		
		self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(12)
		self.pool = nn.MaxPool2d(2, 2)
		
		self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(24)
		
		self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(24)
		
		self.fc1 = nn.Linear(24*10*10, 10)

	def forward(self, input):
		output = F.relu(self.bn1(self.conv1(input)))
		output = F.relu(self.bn2(self.conv2(output)))
		output = self.pool(output)
		output = F.relu(self.bn3(self.conv3(output)))
		output = F.relu(self.bn4(self.conv4(output)))
		output = output.view(-1, 24*10*10)
		output = self.fc1(output)

		return output

