import os
import torch
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.autograd import Variable
from client.radam import RAdam
import pdb
import numpy as np
from torchvision.models import resnet101
import torch.nn as nn
import mlflow

class Trainer():

	def __init__(self, rounds=None, client=None, client_model_path=None):
		self.rounds = rounds
		self.client = client
		self.data_path = "./client/processed.pkl"
		self.label_path = "./client/labels.pkl"
		self.client_model_path = client_model_path

	def load_data(self):
		with open(self.data_path, "rb") as file:
			self.data = pickle.load(file)
		file.close()
		with open(self.label_path, "rb") as file:
			self.labels = pickle.load(file)
		file.close()

		image_files = list(self.data.keys())
		train_keys, val_keys = train_test_split(image_files, test_size=0.2, shuffle=True)

		self.train_data = list()
		self.train_labels = list()
		self.val_data = list()
		self.val_labels = list()

		for key in train_keys:
			self.train_data.append(self.data[key].numpy())
			self.train_labels.append(self.labels[key])

		for key in val_keys:
			self.val_data.append(self.data[key].numpy())
			self.val_labels.append(self.labels[key])

	def train(self, BATCH_SIZE=None, EPOCHS=None, IMG_SIZE=None):

		self.model = resnet101(pretrained=None)
		self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.model.last_linear = nn.Sequential(nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
											   nn.Dropout(p=0.25),
											   nn.Linear(in_features=2048, out_features=2048, bias=True),
											   nn.ReLU(),
											   nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
											   nn.Dropout(p=0.5),
											   nn.Linear(in_features=2048, out_features=1, bias=True))

		if os.path.exists(os.path.join(self.client_model_path, "{}.pt".format(self.client))):
			self.model.load_state_dict(torch.load(os.path.join(self.client_model_path, "{}.pt".format(self.client)))['weights'])

		self.model.to("cuda:0")
		self.loss_fn = torch.nn.CrossEntropyLoss()
		#self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
		self.optimizer = RAdam(self.model.parameters(), lr=0.1)

		for epoch in range(EPOCHS):
			start = 0
			finish = len(self.train_data)
			self.model.train()
			self.optimizer.zero_grad()
			counter = 0
			running_loss = 0.0
			running_acc = 0.0
			outputs = list()
			true = list()
			while start<finish:
				to_train = self.train_data[start:start+BATCH_SIZE]
				lab = self.train_labels[start:start+BATCH_SIZE]
				X, y = torch.Tensor(to_train).reshape(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to("cuda:0"), torch.LongTensor(lab).to("cuda:0")
				X, y = Variable(X), Variable(y)
				out = self.model(X)
				loss = self.loss_fn(out, y)
				loss.backward()
				self.optimizer.step()
				running_loss += loss.item()
				out, y = out.detach().cpu().numpy(), y.detach().cpu().numpy()
				for index in range(len(out)):
					outputs.append(np.argmax(out[index]))
					true.append(y[index])
				start += BATCH_SIZE
				counter += 1

			running_loss = running_loss/counter
			correct = 0
			for index in range(len(true)):
				if true[index] == outputs[index]:
					correct += 1
			running_acc = correct/len(true)

			start = 0
			finish = len(self.val_data)
			counter = 0
			val_loss = 0.0
			val_acc = 0.0
			outputs = list()
			true = list()

			with torch.no_grad():
				while start<finish:
					to_val = self.val_data[start:start+BATCH_SIZE]
					lab = self.val_labels[start:start+BATCH_SIZE]
					X, y = torch.Tensor(to_val).reshape(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to("cuda:0"), torch.LongTensor(lab).to("cuda:0")
					X, y = Variable(X), Variable(y)
					out = self.model(X)
					loss = self.loss_fn(out, y)
					val_loss += loss.item()
					out, y = out.detach().cpu().numpy(), y.detach().cpu().numpy()
					for index in range(len(out)):
						outputs.append(np.argmax(out[index]))
						true.append(y[index])
					start += BATCH_SIZE
					counter += 1

			val_loss = val_loss/counter
			correct = 0
			for index in range(len(true)):
				if true[index] == outputs[index]:
					correct += 1
			val_acc = correct/len(true)

			print("epoch: {} loss: {} val_loss: {} acc: {} val_acc: {}".format(epoch, round(running_loss, 4), round(val_loss, 4), round(running_acc, 4), round(val_acc, 4)))
			torch.cuda.empty_cache()
			mlflow.log_metric("{}_val_loss".format(self.client), val_loss)
			mlflow.log_metric("{}_val_acc".format(self.client), val_acc)
		print()
		self.model.cpu()
		self.model.eval()
		torch.save({"model": self.model, "weights": self.model.state_dict()}, "{}/{}.pt".format(self.client_model_path, self.client))