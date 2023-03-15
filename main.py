# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from models import DeepAutoencoder
from tqdm import tqdm
import train

def plot_reconstructed_original_images(outputs):
	# Plotting is done on a 7x5 subplot
	# Plotting the reconstructed images

	# Initializing subplot counter
	counter = 1

	# Plotting reconstructions
	# for epochs = [1, 5, 10, 50, 100]
	epochs_list = [1, 5, 10, 50, 100]

	# Iterating over specified epochs
	for val in epochs_list:
		
		# Extracting recorded information
		temp = outputs[val]['out'].detach().numpy()
		title_text = f"Epoch = {val}"
		
		# Plotting first five images of the last batch
		for idx in range(5):
			plt.subplot(7, 5, counter)
			plt.title(title_text)
			plt.imshow(temp[idx].reshape(28,28), cmap= 'gray')
			plt.axis('off')
			
			# Incrementing the subplot counter
			counter+=1

	# Plotting original images

	# Iterating over first five
	# images of the last batch
	for idx in range(5):
		
		# Obtaining image from the dictionary
		val = outputs[10]['img']
		
		# Plotting image
		plt.subplot(7,5,counter)
		plt.imshow(val[idx].reshape(28, 28),
				cmap = 'gray')
		plt.title("Original Image")
		plt.axis('off')
		
		# Incrementing subplot counter
		counter+=1

	plt.tight_layout()
	plt.show()


def plot_images():
	# Dictionary that will store the different
	# images and outputs for various epochs
	outputs = {}

	# Extracting the last batch from the test
	# dataset
	img, _ = list(test_loader)[-1]

	# Reshaping into 1d vector
	img = img.reshape(-1, 28 * 28)

	# Generating output for the obtained
	# batch
	out = model(img)

	# Storing information in dictionary
	outputs['img'] = img
	outputs['out'] = out

	# Plotting reconstructed images
	# Initializing subplot counter
	counter = 1
	val = outputs['out'].detach().numpy()

	# Plotting first 10 images of the batch
	for idx in range(10):
		plt.subplot(2, 10, counter)
		plt.title("Reconstructed \n image")
		plt.imshow(val[idx].reshape(28, 28), cmap='gray')
		plt.axis('off')

		# Incrementing subplot counter
		counter += 1

	# Plotting original images

	# Plotting first 10 images
	for idx in range(10):
		val = outputs['img']
		plt.subplot(2, 10, counter)
		plt.imshow(val[idx].reshape(28, 28), cmap='gray')
		plt.title("Original Image")
		plt.axis('off')

		# Incrementing subplot counter
		counter += 1

	plt.tight_layout()
	plt.show()



if __name__ == "__main__":
	
	PATH = "./model.pth"
	new_model = False
	download = False

	# Downloading the MNIST dataset
	train_dataset = torchvision.datasets.MNIST(
		root="./MNIST/train", train=True,
		transform=torchvision.transforms.ToTensor(),
		download=download)

	test_dataset = torchvision.datasets.MNIST(
		root="./MNIST/test", train=False,
		transform=torchvision.transforms.ToTensor(),
		download=download)

	# Creating Dataloaders from the
	# training and testing dataset
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=256)
	test_loader = torch.utils.data.DataLoader(
		test_dataset, batch_size=256)



	# Instantiating the model and hyperparameters
	if new_model:
		model = DeepAutoencoder()
		num_epochs = 10
		outputs, train_loss = train.train_model(model, train_loader, num_epochs)
		torch.save(model, PATH)
		train.plot_loss(num_epochs, train_loss)
	else:
		model = torch.load(PATH)
		model.eval()

	plot_images()

