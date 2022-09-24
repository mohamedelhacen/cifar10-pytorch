from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
import os
import cv2

import torch
from torchvision.transforms import transforms

from utils import allowed_images

app = Flask(__name__)
app.config['UPLOADS'] = './static'
app.config['ALLOWED_EXTENTIONS'] = ['JPEG', 'JPG', 'PNG']

classes = ['airplane', 'automibile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model = torch.load('model.pth')
model.eval()


@app.route('/', methods=["GET", 'POST'])
def upload_image():

	if request.method == 'POST':
		if request.files:
			image = request.files['file']

			if image.filename == '':
				return redirect(request.url)

			if allowed_images(image.filename, app.config['ALLOWED_EXTENTIONS']):
				
				filename = secure_filename(image.filename)
				image.save(os.path.join(app.config['UPLOADS'], filename))

				return redirect(f'/showing_image/{filename}')

			else:
				return redirect(request.url)
	
	return render_template('upload_images.html') 


@app.route('/showing_image/<image_name>', methods=['GET', 'POST'])
def showing_image(image_name):
	
	if request.method == 'POST':
		image_path = os.path.join(app.config["UPLOADS"], image_name)

		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		img = image.copy()
		img = cv2.resize(img, [32, 32])

		pytorch_img = transformations(img)
		pytorch_img = pytorch_img.unsqueeze(0)

		outputs = model(pytorch_img.cuda())
		prob, predictions = torch.max(outputs, 1)

		prob = prob.item() 
		image_class = classes[predictions.item()]

		return render_template('showing_results.html', image_name=image_name, image_class=image_class, prob= round(prob, 2))

	return render_template('showing_image.html', value=image_name)