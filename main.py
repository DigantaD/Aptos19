import os, sys
import uuid
import uuid
import torch
import torchvision
import cv2
import numpy as np
import mlflow
import pandas as pd
import pdb
from sender.get_data import GetData
from client.train import Trainer
import shutil
from server.aggregate import Aggregate
from server.test import Test
from updatemodels import UpdateModels

DATA_FOLDER = "./data"
LABELS = [0, 1, 2, 3, 4]
CLIENTS = os.listdir(DATA_FOLDER)
CLIENTS.remove("test")

model_folders = ['client_models', 'federated_models']
for folder in model_folders:
	if os.path.isdir(folder):
		shutil.rmtree(folder, ignore_errors=True)
	os.mkdir(folder)

CLIENT_MODELS_PATH = "client_models"
FED_MODELS_PATH = "federated_models"
TEST_DATA_PATH = "data/test"

print("===============================DATA DISTRIBUTION==============================\n")

for client in CLIENTS:
	dict1 = dict()
	for label in LABELS:
		dict1[label] = 0
	images = os.listdir(os.path.join(DATA_FOLDER, client))
	labels = pd.read_csv(os.path.join(DATA_FOLDER, "{}/train_labels.csv".format(client)), index_col=False)
	for image in images:
		if image.endswith(".png") or image.endswith("jpg") or image.endswith("jpeg"):
			image = image[:-4]
			label = labels['label'][labels['image'].tolist().index(image)]
			dict1[label] += 1
	print("{}: {}".format(client, dict1))
print()

rounds = 0
flag = 0
images_per_round = 100
start = 0
finish = start + images_per_round

while flag == 0:
	print("Round: {} Range: {}-{}".format(rounds+1, start, finish))
	fetch_obj = GetData(start=start, finish=finish)
	for client in CLIENTS:
		path = os.path.join(DATA_FOLDER, client)
		selected = fetch_obj.fetch(path=os.path.join(DATA_FOLDER, client))
		if len(selected) == 0:
			flag = 1
			break
		print()
		print("Data received for training from {}".format(client))
		print()
		train_obj = Trainer(rounds=rounds, client=client, client_model_path=CLIENT_MODELS_PATH)
		train_obj.load_data()
		train_obj.train(BATCH_SIZE=4, EPOCHS=15, IMG_SIZE=300)

	print("=======================================AGGREGATING MODELS======================================\n")
	agg = Aggregate(CLIENT_MODELS_PATH, FED_MODELS_PATH, CLIENTS)
	agg.aggregate()
	# print("Testing Aggregate Model...")
	# test_obj = Test(fed_model_path=FED_MODELS_PATH, test_data_path=TEST_DATA_PATH)
	# test_obj.fetch()
	# test_obj.evaluate(BATCH_SIZE=4, IMG_SIZE=300)
	print()
	print("Updating Client Models...")
	updt = UpdateModels(client_models_path=CLIENT_MODELS_PATH, fed_model_path=FED_MODELS_PATH)
	updt.update()
	print("Client models updated!")
	print()
	start = finish
	finish += images_per_round
	rounds += 1