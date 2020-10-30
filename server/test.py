import os
import pandas as pd
from sender.preprocess import Preprocess
from tqdm import tqdm
import cv2
from torchvision.models import resnet101
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import mlflow
import pdb

class Test():

	def __init__(self, fed_model_path=None, test_data_path=None):
		self.fed_model_path = fed_model_path
		self.test_data_path = test_data_path

	def fetch(self):
		image_files = os.listdir(self.test_data_path)
		image_files.remove("labels.csv")
		labels = pd.read_csv(os.path.join(self.test_data_path, "labels.csv"))
		preproc = Preprocess(WIDTH=300, HEIGHT=300)
		processed_images = list()
		labels_to_send = list()
		for image in tqdm(image_files):
			file = os.path.join(self.test_data_path, image)
			img = cv2.imread(file)
			img = preproc.preprocess_image(img)
			image = image[:-4]
			label = labels['label'][labels['image'].tolist().index(image)]
			processed_images.append(img)
			labels_to_send.append(label)

		self.test_images = processed_images
		self.test_labels = labels_to_send

		self.test_data = list()
		for img in self.test_images:
			self.test_data.append(img.numpy())

	def evaluate(self, BATCH_SIZE=None, IMG_SIZE=None):
		self.model = resnet101(pretrained=None)
		self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.model.last_linear = nn.Sequential(nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
											   nn.Dropout(p=0.25),
											   nn.Linear(in_features=2048, out_features=2048, bias=True),
											   nn.ReLU(),
											   nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
											   nn.Dropout(p=0.5),
											   nn.Linear(in_features=2048, out_features=1, bias=True))

		self.model.load_state_dict(torch.load(os.path.join(self.fed_model_path, "aggregated_model.pt"))['weights'])
		self.model.to("cuda:0")
		self.model.eval()

		self.loss_fn = nn.CrossEntropyLoss()

		start = 0
		finish = len(self.test_images)
		test_loss = 0.0
		test_acc = 0.0
		counter = 0
		outputs = list()
		true = list()
		with torch.no_grad():
			while start < finish:
				to_test = self.test_data[start:start+BATCH_SIZE]
				lab = self.test_labels[start:start+BATCH_SIZE]
				X, y = torch.Tensor(to_test).reshape(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to("cuda:0"), torch.LongTensor(lab).to("cuda:0")
				X, y = Variable(X), Variable(y)
				out = self.model(X)
				loss = self.loss_fn(out, y)
				test_loss += loss.item()
				out, y = out.detach().cpu().numpy(), y.detach().cpu().numpy()
				for index in range(len(out)):
					outputs.append(np.argmax(out[index]))
					true.append(y[index])
				start += BATCH_SIZE
				counter += 1

			test_loss = test_loss/counter
			correct = 0
			for index in range(len(true)):
				if true[index] == outputs[index]:
					correct += 1
			test_acc = correct/len(true)

			print("aggregate_model_loss: {} aggregate_model_acc: {}".format(round(test_loss, 4), round(test_acc, 4)))
			mlflow.log_metric("aggregate_model_loss", test_loss)
			mlflow.log_metric("aggregate_model_acc", test_acc)