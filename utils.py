import torch
from torch.autograd import Variable

import torchvision

import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

def save_model(model, path):
	torch.save(model, path)


def test_accuracy(model, test_loader):
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


def train(model, train_loader, test_loader, num_epochs, optimizer, loss_fn):
	
	best_accuracy = 0.0
	
	
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
		
		accuracy = test_accuracy(model, test_loader=test_loader)
		print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %d %%'%(accuracy))

		if accuracy > best_accuracy:
			save_model(model, 'model.pth')
			best_accuracy = accuracy


def image_show(img, label=None):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	if label:
		plt.title(label)
	plt.show()


def test_batch(model, test_loader, classes, batch_size):
	images, labels = next(iter(test_loader))

	

	print('Real labels:', ' '.join('%5s' % classes[labels[j]] 
									for j in range(batch_size)))		

	outputs = model(images.cuda())

	_, predictions = torch.max(outputs, 1)
	print('Predicted: ', ' '.join('%5s' % classes[predictions[j]] for j in range(batch_size)))

	image_show(torchvision.utils.make_grid(images))


def test_classes(model, test_loader, number_of_labels, batch_size, classes):
	class_correct = list(0. for i in range(number_of_labels))
	class_total = list(0. for i in range(number_of_labels))

	with torch.no_grad():
		for data in test_loader:
			images, labels = data
			outputs = model(images.to(device))
			_, predictions = torch.max(outputs, 1)
			c = (predictions.cpu() == labels).squeeze()
			for i in range(batch_size):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1
	
	for i in range(number_of_labels):
		print('Accuracy of %5s : %2d %%'%(classes[i], 100*class_correct[i]/class_total[i]))



def allowed_images(filename, extentions):

	if not '.' in filename:
		return False

	ext = filename.rsplit('.', 1)[1]

	if ext.upper() in extentions:
		return True

	else:
		return False