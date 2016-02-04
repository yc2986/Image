# Image Reconstruction
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d, gaussian
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.pyplot import imshow
from PIL import Image
from skimage.restoration import nl_means_denoising

def load(src):
	im = Image.open(src).convert('L')
	width, height = im.size
	arr = np.array(list(im.getdata())).reshape(height, width)
	return np.uint8(arr)

def plot(img):
	imshow(img, cmap = cm.gray)
	plt.show()

def salt_n_pepper(img, prob = 0.05):
	dummy = np.copy(img)
	sp = np.random.randint(1, 101, np.shape(img))
	lower = prob / 2 * 100
	upper = 100 - lower
	# Salt
	dummy[np.where(sp <= lower)] = 0
	# Pepper
	dummy[np.where(sp >= upper)] = 255
	return dummy

def gaussian_add(img, sigma = 5):
	dummy = np.copy(img).astype(float)
	gauss = np.random.normal(0, sigma, np.shape(img))
	# Additive Noise
	dummy = np.round(gauss + dummy)
	# Saturate lower bound
	dummy[np.where(dummy < 0)] = 0
	# Saturate upper bound
	dummy[np.where(dummy > 255)] = 255
	return np.uint8(dummy)

def hist(img, plot = True):
	bins = np.arange(257)
	hist, bins = np.histogram(img, bins = bins)
	if plot == True:
		bins = bins[:-1]
		plt.bar(bins, hist)
		plt.show()
	return bins, hist

def median_filter(img, block_size):
	dummy = np.copy(img)
	height, width = np.shape(img)
	edge = block_size / 2
	pad = np.lib.pad(dummy, [(edge, edge), (edge, edge)], 'edge')
	for i in xrange(edge, edge + height):
		for j in xrange(edge, edge + width):
			ii = i - edge
			jj = j - edge
			dummy[ii, jj] = np.median(pad[i-edge:i+edge+1, j-edge:j+edge+1])
	return np.uint8(dummy)

def blur(img, mode = 'box', block_size = 3):
	# mode = 'box' or 'gaussian' or 'motion'
	dummy = np.copy(img)
	if mode == 'box':
		h = np.ones((block_size, block_size)) / block_size ** 2
	elif mode == 'gaussian':
		h = gaussian(block_size, block_size / 3).reshape(block_size, 1)
		h = np.dot(h, h.transpose())
		h /= np.sum(h)
	elif mode == 'motion':
		h = np.eye(block_size) / block_size
	dummy = convolve2d(dummy, h, mode = 'valid')
	return np.uint8(dummy), h

def wiener_filter(img, kernel, SNR = 0):
	dummy = np.copy(img)
	kernel = np.pad(kernel, [(0, dummy.shape[0] - kernel.shape[0]), (0, dummy.shape[1] - kernel.shape[1])], 'constant')
	# Fourier Transform
	dummy = fft2(dummy)
	kernel = fft2(kernel)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + SNR)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return np.uint8(dummy)

def sharpen(img, kernel):
	dummy = np.copy(img)
	self = np.zeros((3,3))
	self[1,1] = 1
	kernel += self
	dummy = convolve2d(dummy, kernel, mode = 'valid')
	print dummy.max()
	return np.uint8(dummy)

def non_local_filter(img):
	nl_mean = nl_means_denoising(np.copy(img))
	print nl_mean
	return nl_mean

if __name__ == "__main__":

	src = '../lena.tiff'
	img = load(src)
	sp = salt_n_pepper(img)
	re, h = blur(img, mode = 'box', block_size = 11)
	gs = gaussian_add(re, sigma = 50)
	deblur = wiener_filter(re, h, 0.05)
	nl_mean = non_local_filter(re)
	plt.subplot(131)
	imshow(gs, cmap = cm.gray)
	plt.title('Noisy')
	plt.subplot(132)
	imshow(deblur, cmap = cm.gray)
	plt.title('Denoise')
	plt.subplot(133)
	imshow(nl_mean, cmap = cm.gray)
	plt.title('Non local filter')
	plt.tight_layout()
	plt.show()
