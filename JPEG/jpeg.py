# JPEG image compression
import numpy as np
import numpy.fft as fft
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.cm as cm
from scipy.fftpack import dct, idct

def load(src, mode = 'YCbCr'):
	# mode = 'YCbCr' or 'L'
	if mode != 'YCbCr' and mode != 'L' and mode != 'RGB':
		raise ValueError('invalid mode!')
	im = Image.open(src).convert(mode)
	width, height = im.size
	if mode == 'YCbCr':
		arr = np.array(list(im.getdata())).reshape(width, height,3)
	elif mode == 'L':
		arr = np.array(list(im.getdata())).reshape(width, height)
	return np.uint8(arr)

def block(im, block_size):
	size = np.shape(im)
	h, w = size[0:2]
	if h % block_size != 0 or w % block_size != 0:
		raise ValueError('invalid block size!')
	h /= block_size
	w /= block_size
	block_buffer = []
	for i in xrange(h):
		for j in xrange(w):
			if len(size) == 3:
				block_buffer.append(im[i * block_size:(i + 1) * block_size, j * block_size: (j + 1) * block_size, :])
			else:
				block_buffer.append(im[i * block_size:(i + 1) * block_size, j * block_size: (j + 1) * block_size])
	return block_buffer

def deblock(block_buffer, block_size):
	length = np.sqrt(len(block_buffer)).astype(np.int32)
	im = 0
	for i in xrange(0, length):
		tile_base = block_buffer[i * length]
		for j in xrange(1, length):
			tile = block_buffer[i * length + j]
			tile_base = np.concatenate((tile_base, tile), axis = 1)
		if i == 0:
			im = tile_base
		else:
			im = np.concatenate((im, tile_base), axis = 0)
	return np.uint8(im)

def plot(im, mode = 'YCbCr'):
	if mode == 'YCbCr':
		imshow(im)
	elif mode == 'L':
		imshow(im, cmap = cm.Greys)
	plt.show()

def dct2(im):
	return dct(dct(im, axis = 0), axis = 1)

def idct2(im, block_size = 8):
	return idct(idct(im, axis = 0), axis = 1) / ((block_size * 2) ** 2)

def DCT(block_buffer):
	size = len(block_buffer)
	buf = []
	for i in xrange(size):
		buf.append(dct2(np.float32(block_buffer[i])))
	return buf

def IDCT(block_buffer):
	size = len(block_buffer)
	buf = []
	for i in xrange(size):
		buf.append(idct2(np.float32(block_buffer[i])))
	return buf

def DFT(block_buffer):
	size = len(block_buffer)
	buf = []
	for i in xrange(size):
		buf.append(fft.fft2(np.float32(block_buffer[i])))
	return buf

def IDFT(block_buffer):
	size = len(block_buffer)
	buf = []
	for i in xrange(size):
		buf.append(fft.ifft2(np.float32(block_buffer[i])))
	return buf

def jpeg_quantization_table_luma():
	return np.array([[16, 11, 10, 16,  24,  40,  51,  61],
					 [12, 12, 14, 19,  26,  58,  60,  55],
					 [14, 13, 16, 24,  40,  57,  69,  56],
					 [14, 17, 22, 29,  51,  87,  80,  62],
					 [18, 22, 37, 56,  68, 109, 103,  77],
					 [24, 35, 55, 64,  81, 104, 113,  92],
					 [49, 64, 78, 87, 103, 121, 120, 101],
					 [72, 92, 95, 98, 112, 100, 103,  99]]).astype(np.int32)

def jpeg_quantization_table_chroma():
	return np.array([[17, 18, 24, 47, 99, 99, 99, 99],
			         [18, 21, 26, 66, 99, 99, 99, 99],
			         [24, 26, 56, 99, 99, 99, 99, 99],
			         [47, 66, 99, 99, 99, 99, 99, 99],
			         [99, 99, 99, 99, 99, 99, 99, 99],
			         [99, 99, 99, 99, 99, 99, 99, 99],
			         [99, 99, 99, 99, 99, 99, 99, 99],
			         [99, 99, 99, 99, 99, 99, 99, 99]]).astype(np.int32)

def quantize(block_buffer, mode = 'YCbCr', luma_scale = 1, chroma_scale = 1):
	# mode = 'YCbCR' or 'L'
	if mode != 'YCbCr' and mode != 'L':
		raise ValueError('invalid mode!')
	block_buffer = np.float32(block_buffer)
	luma = jpeg_quantization_table_luma() * luma_scale
	chroma = jpeg_quantization_table_chroma() * chroma_scale
	size = len(block_buffer)
	for i in xrange(size):
		block = block_buffer[i]
		if mode == 'YCbCr':
			block[:,:,0] = np.round(block[:,:,0] / luma) * luma
			block[:,:,1] = np.round(block[:,:,1] / chroma) * chroma
			block[:,:,2] = np.round(block[:,:,2] / chroma) * chroma
		else:
			block = np.round(block / luma) * luma
		block_buffer[i] = block
	return np.int32(block_buffer)

def rgb2ycbcr(im):
	xform = np.array([[.299, .587, .144],[-.168, -.331, .5],[.5, -.419, -.0813]])
	ycbcr = im.dot(xform.T)
	ycbcr[:,:,[1,2]] += 128
	return np.float32(ycbcr)

def ycbcr2rgb(im):
	xform = np.array([[1, 0, 1.402], [1, -0.344, -.714], [1, 1.772, 0]])
	rgb = im.astype(np.float)
	rgb[:,:,[1,2]] -= 128
	return np.uint8(rgb.dot(xform.T))

def predictor(im, predict_mode = 0):
	# predict_mode = 0 using (-1, 0)
	# predict_mode = 1 using (0, -1)
	# predict_mode = 2 using (-1,-1)
	if predict_mode < 0 or predict_mode > 2:
		raise ValueError('invalid predictor!')
	err = np.empty_like(im).astype(np.int32)
	h, w = np.shape(im)
	im_pad = np.lib.pad(im, [(1,0), (1,0)], mode = 'edge').astype(np.int32)
	for i in xrange(1, h + 1):
		for j in xrange(1, w + 1):
			ii = i - 1
			jj = j - 1
			if predict_mode == 0:
				predict = im_pad[ii, j]
			elif predict_mode == 1:
				predict = im_pad[i, jj]
			else:
				predict = (im_pad[ii,jj] + im_pad[ii,j] + im_pad[i,jj]) / 3
			err[ii, jj] = predict - im_pad[i, j]
	return err

def entropy(hist):
	p = np.float32(hist) / np.sum(hist)
	p = p[np.where(p != 0)]
	e = -np.sum(p * np.log2(p))
	return e

def JPEG(src, mode = 'YCbCr', transform = 'DCT', draw = True, luma_scale = 1, chroma_scale = 1):
	# mode = 'YCbCr' or 'L'
	# transform = 'DCT' or 'DFT' or 'None'
	block_size = 8
	img = load(src, mode)
	#img = rgb2ycbcr(img)
	block_buf = block(img, block_size)
	if transform == 'DCT':
		block_buf = DCT(block_buf)
	elif transform == 'DFT':
		block_buf = DFT(block_buf)
	elif transform == 'None': 1
	else:
		raise ValueError('invalid transform')

	block_buf = quantize(block_buf, mode, chroma_scale = 1)

	if transform == 'DCT':
		block_buf = IDCT(block_buf)
	elif transform == 'DFT':
		block_buf = IDFT(block_buf)

	img = deblock(block_buf, block_size)

	if mode == 'YCbCr':
		img = ycbcr2rgb(img)
	elif mode == 'L':
		img = 255 - img

	if draw:
		plot(img, mode)
	return img

def JPEG_LS(src, mode = 'YCbCr', predict_mode = 0, draw_hist = True):
	img = load(src, mode)
	err = np.empty_like(img).astype(np.int32)
	hist = 0
	e = 0
	bins = np.arange(511) - 255
	if mode == 'YCbCr':
		err[:,:,0] = predictor(img[:,:,0], predict_mode)
		err[:,:,1] = predictor(img[:,:,1], predict_mode)
		err[:,:,2] = predictor(img[:,:,2], predict_mode)
		hist_y, lower_edges = np.histogram(err[:,:,0], bins = np.arange(512) - 255)
		hist_cb, lower_edges = np.histogram(err[:,:,1], bins = np.arange(512) - 255)
		hist_cr, lower_edges = np.histogram(err[:,:,2], bins = np.arange(512) - 255)
		hist = [hist_y, hist_cb, hist_cr]
		e = [entropy(hist_y), entropy(hist_cb), entropy(hist_cr)]
		if draw_hist == True:
			plt.figure()
			plt.subplot(311)
			plt.bar(bins, hist_y)
			plt.title('Y')
			plt.subplot(312)
			plt.bar(bins, hist_cb)
			plt.title('Cb')
			plt.subplot(313)
			plt.bar(bins, hist_cr)
			plt.show()
	elif mode == 'L':
		err = predictor(img, predict_mode)
		hist, lower_edges = np.histogram(err, bins = np.arange(512) - 255)
		e = entropy(hist)
		if draw_hist == True:
			plt.figure()
			plt.bar(bins, hist)
			plt.title('Grayscale')
			plt.show()
	return hist, e

if __name__ == "__main__":
	# JPEG compression
	src = '../lena.tiff'
	mode = 'YCbCr'
	transform = 'DCT'
	img_compress = JPEG(src, mode, transform)

	# Lossless predictor
	src = '../lena.tiff'
	mode = 'YCbCr'
	predict_mode = 2
	hist, entro = JPEG_LS(src, predict_mode = predict_mode)
	print 'Entropy: ', entro