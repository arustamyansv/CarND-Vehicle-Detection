import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from glob import glob

from skimage.feature import hog
from time import time

UCONFIG = {
	'color': cv2.COLOR_BGR2YCrCb,

	'hog': {
		'orientations': 9,
		'pixels_per_cell': (8, 8),
		'cells_per_block': (2, 2),
		'block_norm': 'L2-Hys'
	},

	'hist': {
		'bins': 32,
	},

	'spatial': {
    	'shape': (32, 32)
	}

}

def extract_features_hog_channel(img, ch, feature_vector=True, debug=False):
	"""
	Extract features for single channel using HOG algorythm.
	"""

	global UCONFIG

	result = hog(
		img[:, :, ch],
        feature_vector=feature_vector,
        visualise=debug,
        **UCONFIG['hog']
	)

	if debug:
		draw (result[1], False)
		return result[0]
    
	return result

def extract_features_hog(img, feature_vector=True, crop=None, scale_factor=None, debug=False):
	"""
	Extract features for HOG using all channels. 
	In case we are running feature extraction for the big picture with overlapping windows and scaling
	it will allow to crop and scale area appropriately.
	"""

	if crop is not None and scale_factor is not None:
		h, w, c = img.shape

		img = cv2.resize(img, (w//scale_factor, h//scale_factor))
		img = img[crop[0]//scale_factor:, crop[1]//scale_factor:,]

	# run hog for each channel separatelly 
	result = [
		extract_features_hog_channel(img, 0, feature_vector, debug=debug),
		extract_features_hog_channel(img, 1, feature_vector, debug=debug),
		extract_features_hog_channel(img, 2, feature_vector, debug=debug)
	]

	# in case we need just a feature vector 
	if feature_vector:
		return np.ravel(result)

	return result

def extract_features_hog_scaled(hog, window_options, box):
	"""
	Extract features from the prepared HOG features.
	Utilised to speed up detection process.
	"""

	global UCONFIG

	ppc = np.array(UCONFIG['hog']['pixels_per_cell'])
	cpb = np.array(UCONFIG['hog']['cells_per_block'])

	scale_factor = window_options['scale_factor']
	x_shift = window_options['x_start_stop'][0]
	y_shift = window_options['y_start_stop'][0]

	# make shifts according to crop
	start = np.array([box[0][0] - x_shift, box[0][1] - y_shift])
	stop  = np.array([box[1][0] - x_shift, box[1][1] - y_shift])

	# rescale points 
	start = ((start / scale_factor // ppc) - (cpb - 1) + 1).astype('int') 
	stop  = ((stop  / scale_factor // ppc) - (cpb - 1)).astype('int') 

	subsamples = []
	for channel in hog:
		subsamples.append(np.ravel(channel[start[1]:stop[1], start[0]:stop[0]]))
	
	return np.ravel(subsamples)

def extract_features_hist(img, debug=False):
	"""
	Extract features list using histogram
	"""

	global UCONFIG

	bins = UCONFIG['hist']['bins']

	ch1 = np.histogram(img[:, :, 0], bins=bins)
	ch2 = np.histogram(img[:, :, 1], bins=bins)
	ch3 = np.histogram(img[:, :, 2], bins=bins)

	if debug:
		draw_color_hist(ch1, ch2, ch3)

	return np.concatenate((ch1[0], ch2[0], ch3[0]))

def extract_features_spatial(img, debug=False):
	"""
	Extract spatial features from the image
	"""

	global UCONFIG

	result = cv2.resize(img, UCONFIG['spatial']['shape'])
	if debug:
		draw(result)

	return result.ravel()

def extract_features(img, debug=False):
	"""
	Extract features from all methods if they are specified in the configuration
	"""

	global UCONFIG

	# convert to YCrCb colorspace
	img = cv2.cvtColor(img, UCONFIG['color'])
	features = []

	# get hog features
	if 'hog' in UCONFIG:
		features.append(extract_features_hog(img, debug=debug))

	# load histogram features
	if 'hist' in UCONFIG:
		features.append(extract_features_hist(img, debug=debug))

	# get spatiual features
	if 'spatial' in UCONFIG:
		features.append(extract_features_spatial(img, debug=debug))

	# combine and return them
	return np.concatenate(features)

def load_images(folders):
    """
    Loads list of image paths from filesystem.
    """

    paths = []
    for folder in folders:
        paths += glob('{}/*'.format(folder))

    return paths

def load_image(path):
	"""
	Loads image content from filesystem.
	"""
	
	return cv2.imread(path)

def draw_boxes(img, boxes, color=(0, 0, 255), thickness=3):
    """
    Draw boxes on the image.
    """

    # Make a copy of the image just in case
    result = np.copy(img)
    
    # Iterate through the bounding boxes
    for box in boxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(result, box[0], box[1], color, thickness)
        
    # Return the image copy with boxes drawn
    return result

def draw(img, bgr=True):
    """
    Draw image on the screen.
    """

    if bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pass
        
    plt.imshow(img)
    plt.show()

def draw_color_hist(channel1_hist, channel2_hist, channel3_hist):
	"""
	Draw intermediate result for histogram feature extraction.
	"""
	
	# Generating bin centers
	bin_edges = channel1_hist[1]
	bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

	fig = plt.figure(figsize=(12,3))
	plt.subplot(131)
	plt.bar(bin_centers, channel1_hist[0])
	plt.xlim(0, 256)
	plt.title('R Histogram')
	plt.subplot(132)
	plt.bar(bin_centers, channel2_hist[0])
	plt.xlim(0, 256)
	plt.title('G Histogram')
	plt.subplot(133)
	plt.bar(bin_centers, channel3_hist[0])
	plt.xlim(0, 256)
	plt.title('B Histogram')
	plt.show()

def rescale_box(box, shift = 65):
	"""
	Method will get a box treated as hit and will rescale it to fixed square.
	"""

	# get centers of the coordinates so we will know where should we start from new square
	center_y = math.floor((box[1][0]+box[0][0])/2)
	center_x = math.floor((box[1][1]+box[0][1])/2)

	new_box = (
		(center_y - shift, center_x - shift),
		(center_y + shift, center_x + shift)
	)

	return new_box

def get_square(box):
	"""
	Calculate square of the box.
	"""

	return (box[1][0]-box[0][0]) * (box[1][1]-box[0][1])

def ptitle(msg):
	"""
	Method will print title-style to output
	"""

	print ()
	print ('==============================================')
	print (msg)
	print ()

if __name__ == "__main__":
	ptitle('Runing features extraction...')

	path = 'test_images/test1.jpg'
	img = load_image(path)
	features = extract_features(img, debug=True)
