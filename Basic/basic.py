import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import imshow

"""
Write a computer program capable of reducing the number of intensity 
levels in an image from 256 to 2, in integer powers of 2. 
The desired number of intensity levels needs to be a variable input 
to your program.
"""

def load(src):
	im = Image.open(src).convert('L')
	width, height = im.size
	greyscale_map = np.array(list(im.getdata())).reshape((width, height))
	return np.uint8(greyscale_map)

def quantize(im, scale):
	return im / scale * scale

def plot(im):
	imshow(im, cmap = cm.gray)
	plt.show()

def blur(im, neighbor):
	h, w = np.shape(im)
	pad = (neighbor - 1) / 2
	im = np.lib.pad(im, [(pad, pad), (pad, pad)], 'constant', constant_values = 128)
	for i in xrange(pad, h + pad):
		for j in xrange(pad, w + pad):
			im[i, j] = np.mean(im[i - pad:i + pad, j - pad:j + pad])
	return im[pad:pad + h, pad:pad + w]

def rotate(im, degree):
	img = Image.fromarray(im)
	return img.rotate(degree)

def average(im, block_size):
	h, w = np.shape(im)
	pad = (block_size - 1) / 2
	im = np.lib.pad(im, [(pad, pad), (pad, pad)], 'constant', constant_values = 128)
	for i in xrange(pad, h + pad, block_size):
		for j in xrange(pad, w + pad, block_size):
			im[i - pad:i + pad, j - pad:j + pad] = np.mean(im[i - pad:i + pad, j - pad:j + pad])
	return im[pad:pad + h, pad:pad + w]

if __name__ == "__main__":
	src = '../lena.tiff'
	lena = load(src)
	quantize = quantize(lena, 4)
	blur = blur(lena,19)
	rotate = rotate(lena, 45)
	mean = average(lena, 16)
	plt.figure()
	plt.subplot(221)
	imshow(quantize, cmap = cm.Greys)
	plt.title('Quantize')
	plt.subplot(222)
	imshow(blur, cmap = cm.Greys)
	plt.title('Blur')
	plt.subplot(223)
	imshow(rotate, cmap = cm.Greys)
	plt.title('45 Rotate')
	plt.subplot(224)
	imshow(mean, cmap = cm.Greys)
	plt.title('Average')
	plt.tight_layout()
	plt.show()