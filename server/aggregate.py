import os
import torch
from torchvision.models import resnet101
import torch.nn as nn

class Aggregate():

	def __init__(self, client_model_path=None, fed_model_path=None, clients=None):
		self.client_model_path = client_model_path
		self.fed_model_path = fed_model_path
		self.clients = clients

	def aggregate(self):
		
		models = os.listdir(self.client_model_path)
		agg_model = resnet101(pretrained=None)
		agg_model.avg_pool = nn.AdaptiveAvgPool2d(1)
		agg_model.last_linear = nn.Sequential(nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
											  nn.Dropout(p=0.25),
											  nn.Linear(in_features=2048, out_features=2048, bias=True),
											  nn.ReLU(),
											  nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
											  nn.Dropout(p=0.5),
											  nn.Linear(in_features=2048, out_features=1, bias=True))

		for layer in agg_model.state_dict().keys():
			done = 0
			for model in models:
				weight = torch.load(os.path.join(self.client_model_path, model))['weights'][layer]
				if done == 0:
					agg_model.state_dict()[layer] = weight
				else:
					agg_model.state_dict()[layer] += (1/len(models))*weight

		torch.save({"model": agg_model, "weights": agg_model.state_dict()}, "{}/aggregated_model.pt".format(self.fed_model_path))