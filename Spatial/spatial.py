# Spatial Filtering
import numpy as np
import numpy as np
import scipy
import scipy.signal
from skimage.restoration import nl_means_denoising
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.cm as cm

def load(src):
	im = Image.open(src).convert('L')
	width, height = im.size
	arr = np.array(list(im.getdata())).reshape(height, width)
	return np.uint8(arr)

def plot(img):
	imshow(img, cmap = cm.gray)
	plt.show()

def histo_eq(img, plot = False):
	bins = np.arange(257)
	hist, lower = np.histogram(img, bins = bins)
	p = np.float32(hist) / np.size(img)
	cdf = np.cumsum(p)
	dictionary = np.uint8(np.round(255 * cdf))
	process = dictionary[img]
	if plot == True:
		x = np.arange(256)
		plt.figure()
		plt.subplot(211)
		plt.bar(x, hist)
		hist, lower = np.histogram(process, bins = bins)
		plt.subplot(212)
		plt.bar(x, hist)
		plt.tight_layout()
		plt.show()
	return process

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

def sp_noise(img, prob = 0.05):
	# Generate Salt & Pepper noise based on the probability provided
	dummy = np.copy(img)
	pool = np.random.randint(1, 100, np.size(img)).reshape(np.shape(img))
	lower = prob / 2 * 100
	upper = 100 - lower
	# Salt
	dummy[np.where(pool <= lower)] = 0
	# Pepper
	dummy[np.where(pool >= upper)] = 255
	return dummy

def random_noise(img, N):
	img_noisy = []
	for i in xrange(N):
		noise = sp_noise(img, 0.5)
		img_noisy.append(noise)
	return img_noisy

def random_filter(img, N, plot = True):
	img_noisy = np.array(random_noise(img, N))
	average = np.uint8(np.average(img_noisy, axis = 0))
	if plot == True:
		plt.figure()
		plt.subplot(211)
		imshow(img_noisy[0,:,:], cmap = cm.gray)
		plt.title('noisy')
		plt.subplot(212)
		imshow(average, cmap = cm.gray)
		plt.title('average')
		plt.show()
	return average

def edge_detect(img, laplacian, threshold = 30):
	dummy = np.copy(img)
	dummy = scipy.signal.convolve2d(dummy, laplacian, mode = "same")
	edge = np.zeros_like(img)
	edge[np.where(dummy >= threshold)] = 255
	return edge

if __name__ == "__main__":

	src = '../lena.tiff'
	src_dark = '../lena_dark.png'
	img1 = load(src)
	img2 = load(src_dark)
	laplacian = np.array([[ 0,-1, 0],
						  [-1, 4,-1],
						  [ 0,-1, 0]])
	he = histo_eq(img2)
	sp = sp_noise(img1)
	median = median_filter(img1, 3)
	non_local = nl_means_denoising(img1)
	average = random_filter(img1, 500, plot = False)
	edge = edge_detect(img1, laplacian)

	plt.figure()
	plt.subplot(241)
	plt.title('Dark Lena')
	imshow(img2, cmap = cm.gray)
	plt.subplot(242)
	plt.title('Hist Equal')
	imshow(he, cmap = cm.gray)
	plt.subplot(243)
	plt.title('Salt & Pepper')
	imshow(sp, cmap = cm.gray)
	plt.subplot(244)
	plt.title('Median Filter')
	imshow(median, cmap = cm.gray)
	plt.subplot(245)
	plt.title('Non Local Mean')
	imshow(non_local, cmap = cm.gray)
	plt.subplot(246)
	plt.title('500 Average')
	imshow(average, cmap = cm.gray)
	plt.subplot(247)
	plt.title('Lena')
	plt.imshow(img1, cmap = cm.gray)
	plt.subplot(248)
	plt.title('Edge Detect')
	plt.imshow(edge, cmap = cm.gray)
	plt.tight_layout()
	plt.show()