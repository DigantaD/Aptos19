from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

class Preprocess():

	def __init__(self, WIDTH, HEIGHT):
		self.WIDTH = WIDTH
		self.HEIGHT = HEIGHT
		self.transform = transforms.Compose([
				transforms.Resize((HEIGHT, WIDTH)),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])

	def preprocess_image(self, image):
		self.image = image
		self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
		self.image = Image.fromarray(self.image)
		self.image = self.transform(self.image)
		return self.image