# example of calculating the frechet inception distance in Keras
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
# from skimage.transform import resize
from scipy.misc import imread, imresize, imsave

import cv2
import tensorflow as tf


# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = imresize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


if __name__ == "__main__":
    # prepare the inception v3 model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # TODO: fix this bug that isv3 not working on my computer.
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    # define two fake collections of images
    im1 = np.array([cv2.imread(r"./save/eg-l1-original/test/input.png")])
    im2 = np.array([cv2.imread(r"./save/eg-l1-original/test/50.png")])
    print('Prepared', im1.shape, im2.shape)
    # convert integer to floating point values
    images1 = im1.astype('float32')
    images2 = im2.astype('float32')
    # resize images
    images1 = scale_images(images1, (299,299,3))
    images2 = scale_images(images2, (299,299,3))
    print('Scaled', images1.shape, images2.shape)
    # pre-process images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    # fid between images1 and images1
    fid = calculate_fid(model, images1, images1)
    print('FID (same): %.3f' % fid)
    # fid between images1 and images2
    fid = calculate_fid(model, images1, images2)
    print('FID (different): %.3f' % fid)

