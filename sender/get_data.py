import cv2
import os
import random
from sender.preprocess import Preprocess
import pandas as pd
import shutil
import pickle
from tqdm import tqdm

class GetData():

	def __init__(self, start=None, finish=None):
		self.start = start
		self.finish = finish

	def fetch(self, path=None):
		files = os.listdir(path)
		labels = pd.read_csv(os.path.join(path, "train_labels.csv"))
		selected = files[self.start:self.finish]
		preproc = Preprocess(WIDTH=300, HEIGHT=300)
		processed_images = dict()
		labels_to_send = dict()
		if "train_labels.csv" in selected:
			selected.remove("train_labels.csv")
		for image in tqdm(selected):
			file = os.path.join(path, image)
			img = cv2.imread(file)
			img = preproc.preprocess_image(img)
			image = image[:-4]
			label = labels['label'][labels['image'].tolist().index(image)]
			processed_images[file] = img
			labels_to_send[file] = label
		with open("./sender/processed.pkl", "wb") as file:
			pickle.dump(processed_images, file)
		file.close()
		with open("./sender/labels.pkl", "wb") as file:
			pickle.dump(labels_to_send, file)
		file.close()
		shutil.move("./sender/processed.pkl", "./client/processed.pkl")
		shutil.move("./sender/labels.pkl", "./client/labels.pkl")
		return selected