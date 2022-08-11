# 50000 training images and 10000 test images with 3x32x32 shape
# 10 classes: {'airplane':0, 'automibile':1, 'bird':2, 'cat':3, 'deer':4, 
# 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

transformations = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 16
number_of_labels = 10

train_set = CIFAR10(root='./data', train=True, transform=transformations, download=True)
test_set = CIFAR10(root='./data', train=False, transform=transformations, download=True)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ['airplane', 'automibile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


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


model = Network()

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

def save_model():
	path = './output_model.pth'
	torch.save(model.state_dict(), path)

def test_accuracy():
	model.eval()
	accuracy = 0.0
	total = 0.0

	with torch.no_grad():
		for data in test_loader:
			images, labels = data
			images = Variable(images.to(device))
			labels = Variable(labels.to(device))
			outputs = model(images)
			_, predictions = torch.max(outputs.data, 1)
			total += labels.size(0)
			accuracy += (predictions == labels).sum().item()

	accuracy = 100 * accuracy / total
	return accuracy

def train(num_epochs):
	global device
	best_accuracy = 0.0
	
	device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
	print("The model will be running on ", device, ' device')
	model.to(device)

	for epoch in range(num_epochs):
		running_loss = 0.0
		running_accuracy = 0.0

		for i, (images, labels) in enumerate(train_loader, 0):

			images = Variable(images.to(device))
			labels = Variable(labels.to(device))

			optimizer.zero_grad()

			outputs = model(images)
			loss = loss_fn(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if i % 1000 == 999:
				print('[%d, %5d] loss: %.3f'%(epoch + 1, i+1, running_loss/1000))
				running_loss = 0.0
		
		accuracy = test_accuracy()
		print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %d %%'%(accuracy))

		if accuracy > best_accuracy:
			save_model()
			best_accuracy = accuracy


def image_show(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

def test_batch():
	images, labels = next(iter(test_loader))

	

	print('Real labels:', ' '.join('%5s' % classes[labels[j]] 
									for j in range(batch_size)))		

	outputs = model(images)

	_, predictions = torch.max(outputs, 1)
	print('Predicted: ', ' '.join('%5s' % classes[predictions[j]] for j in range(batch_size)))

	image_show(torchvision.utils.make_grid(images))

def test_classes():
	class_correct = list(0. for i in range(number_of_labels))
	class_total = list(0. for i in range(number_of_labels))

	with torch.no_grad():
		for data in test_loader:
			images, labels = data
			outputs = model(images)
			_, predictions = torch.max(outputs, 1)
			c = (predictions == labels).squeeze()
			for i in range(batch_size):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1
	
	for i in range(number_of_labels):
		print('Accuracy of %5s : %2d %%'%(classes[i], 100*class_correct[i]/class_total[i]))

if __name__ == "__main__":

	train(5)
	print("[INFO]: Finishin training")

	test_accuracy()
	
	model = Network()
	path = 'output_model.pth'
	model.load_state_dict(torch.load(path))

	test_classes()
	test_batch()