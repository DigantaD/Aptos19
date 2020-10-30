import os
from torchvision.models import resnet101
import torch
import torch.nn as nn
from tqdm import tqdm

class UpdateModels():

	def __init__(self, client_models_path=None, fed_model_path=None):
		self.client_models_path = client_models_path
		self.fed_model_path = fed_model_path

	def update(self):
		agg_model = resnet101(pretrained=None)
		agg_model.avg_pool = nn.AdaptiveAvgPool2d(1)
		agg_model.last_linear = nn.Sequential(nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
											  nn.Dropout(p=0.25),
											  nn.Linear(in_features=2048, out_features=2048, bias=True),
											  nn.ReLU(),
											  nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
											  nn.Dropout(p=0.5),
											  nn.Linear(in_features=2048, out_features=1, bias=True))
		agg_model.load_state_dict(torch.load(os.path.join(self.fed_model_path, "aggregated_model.pt"))['weights'])

		client_model = resnet101(pretrained=None)
		client_model.avg_pool = nn.AdaptiveAvgPool2d(1)
		client_model.last_linear = nn.Sequential(nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
											  	 nn.Dropout(p=0.25),
											  	 nn.Linear(in_features=2048, out_features=2048, bias=True),
											  	 nn.ReLU(),
											  	 nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
											  	 nn.Dropout(p=0.5),
											  	 nn.Linear(in_features=2048, out_features=1, bias=True))

		files = os.listdir(self.client_models_path)

		for file in tqdm(files):

			client_model.load_state_dict(torch.load(os.path.join(self.client_models_path, "{}".format(file)))['weights'])

			new_weights = client_model.state_dict()

			for layer in client_model.state_dict().keys():
				if layer.endswith('weight'):
					new_weights[layer] = 0.7*client_model.state_dict()[layer] + 0.3*agg_model.state_dict()[layer]

			torch.save({"model": client_model, "weights": new_weights}, "{}/{}".format(self.client_models_path, file))