import numpy as np
import cv2
from scipy.ndimage.measurements import label
from sklearn.externals import joblib
from time import time
from moviepy.editor import VideoFileClip

from utils import *
from windows import *

from model import MCONFIG

DCONFIG = {
	'heatmap': {
		'heatlog_size': 3,
		'mode': 'default',
		# 'mode': 'sum',
		'threshould': 1
	},
	'small_boxes_filter': 3
}
DCACHE = {}

def fit_boxes(img, clf, scaler):
	"""
	Method is responsible for generating predictions on the image.
	"""

	global DCACHE
	global WCONFIG

	# load cached windows list. No need to re-generate it each time
	if 'windows' not in DCACHE:
		DCACHE['windows'] = generate_windows()
	
	windows = DCACHE['windows']

	# get hog for the whole area of interes and then subsample small boxes as needed
	img_ycrcb = cv2.cvtColor(img, UCONFIG['color'])

	# aggregate options just to get them more comfortably
	conf = {options['scale_factor']:options for options in WCONFIG['windows']}

	# iterate through the windows and generate list of subsamples for each scale
	hogs = {}
	for options in WCONFIG['windows']:
		crop = (options['y_start_stop'][0], options['x_start_stop'][0])

		features_hog = extract_features_hog(img_ycrcb, False, crop, options['scale_factor'])
		
		# save it for main loop
		hogs[options['scale_factor']] = {
			'hog': features_hog, 
			'window_options': options
		}

	# ok let's walk through the windowses and try to fit our cars
	boxes = []
	for (pt1, pt2, scale_factor) in windows:

		# obtain box data for identification
		_img = img_ycrcb[pt1[1]:pt2[1], pt1[0]:pt2[0]]

		# preprocess data
		_img = cv2.resize(_img, (64, 64))

		# get hog features specificaly for this box
		params = hogs[scale_factor]
		params['box'] = (pt1, pt2)
		
		# aggregate all features - one we got from HOG and other generated per image
		features = np.concatenate([
			extract_features_hog_scaled(**params),
			extract_features_hist(_img),
			extract_features_spatial(_img)
		])

		# scale features 	# rescale strong boxes
	# boxes = [rescale_box(box) same way we did during training
		features = scaler.transform(features.reshape(1, -1))
		
		# and finally check if this box contain a car or not
		if clf.predict(features):	
			boxes.append((pt1, pt2))

	return boxes

def compose_heatmap(img, boxes):
	"""
	Method will process heatmap logic. Including heatmap log caching and values averaging.
	"""

	global DCONFIG
	global DCACHE

	heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)

	for (start, stop) in boxes:
		heatmap[start[1]:stop[1], start[0]:stop[0]] += 1

    # get last 3 heatmaps from the history and include in labeled hitmap calculation
	if 'heatlog' not in DCACHE:
		DCACHE['heatlog'] = []

	config = DCONFIG['heatmap']

	DCACHE['heatlog'].append(heatmap)
	heatlog = DCACHE['heatlog'][-config['heatlog_size']:]

	if config['mode'] == 'sum':
		heatmap = np.sum(heatlog, axis=0)

	else:
		heatmap = np.vstack(heatlog)

	# apply threshoulding for heatmap
	heatmap[heatmap <= config['threshould']] = 0

	return heatmap

def detect(img, clf, scaler):
	"""
	Method will run detection on image with given classifier and scaler
	"""

	global DCACHE
	global DCONFIG

	# try to find something on windowses
	boxes = fit_boxes(img, clf, scaler)

	# generate heatmap based on predictions
	heatmap = compose_heatmap(img, boxes)

	# distinguish car heats between each other
	labels = label(heatmap)
	print (labels)
	draw(labels[0])
	# generate final boxes for the cars
	boxes = []
	for i in range(1, labels[1]+1):

		# Find pixels with each car_number label value
		nonzero = (labels[0] == i).nonzero()

		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])

		# Define a bounding box based on min/max x and y
		boxes.append((
			(np.min(nonzerox), np.min(nonzeroy)), 
			(np.max(nonzerox), np.max(nonzeroy))
		))

	# remove week boxes
	# boxes = sorted(boxes, key=get_square)[-DCONFIG['small_boxes_filter']:]

	# rescale strong boxes
	# boxes = [rescale_box(box) for box in boxes]

	img = draw_boxes(img, boxes)

	return img

if __name__ == "__main__":
	ptitle('Trying to detect cars on test image...')

	path = 'test_images/test1.jpg'
	img = load_image(path)
	print ('smthng')
	clf, scaler = joblib.load(MCONFIG['model_file'])

	img = detect(img, clf, scaler)
	draw(img)

#if __name__ == "__main__":
#	path = 'model.pkl'

#	clf, scaler = joblib.load(path)

#	vpath = 'project_video.mp4'
	#vpath = 'test_video.mp4'
#	clip1 = VideoFileClip(vpath) #.subclip(10, 40)
#	white_clip = clip1.fl_image(lambda img: detect(img, clf, scaler))
#	white_clip.write_videofile("project_video_complete.mp4", audio=False)
