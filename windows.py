from utils import *
import numpy as np

WCONFIG = {
	'windows': [
		{
			'x_start_stop': (700, 1280), 
			'y_start_stop': (400, 590), 
			'window_size' : 64,
			'step_size'   : 16,
			'scale_factor': 1
		},
		{
			'x_start_stop': (700, 1280), 
			'y_start_stop': (400, 650), 
			'window_size' : 64,
			'step_size'   : 32 ,
			'scale_factor': 2
		}
		# ,
		# {
		# 	'x_start_stop': (720, 1280), 
		# 	'y_start_stop': (380, 680), 
		# 	'window_size' : 64,
		# 	'step_size'   : 16,
		# 	'scale_factor': 2
		# }	
	]
}

def slide_window(x_start_stop, y_start_stop, window_size, step_size, scale_factor):
	"""
	Method will generate windows in the choosen area of interests.
	"""

	# rescale accotring to factor
	window_size = window_size*scale_factor

	# calculate number of boxes for this window in both x and y directions
	aoi_shape = np.array([y_start_stop[1] - y_start_stop[0], x_start_stop[1] - x_start_stop[0]])
	counts =  (((aoi_shape - window_size)//step_size) + 1).astype('int')
	
	# walk through calculations and generate boxes
	windows = []
	for iy in range(counts[0]):
		for ix in range(counts[1]):
			y_start = y_start_stop[0] + iy*step_size
			x_start = x_start_stop[0] + ix*step_size

			windows.append((
				(x_start, y_start), 
				(x_start + window_size, y_start + window_size),
				scale_factor
			))

	return windows

def generate_windows():
	"""
	Main method that will generate windows and then stack them together.
	"""

	global WCONFIG
	return np.vstack([slide_window(**cfg) for cfg in WCONFIG['windows']])


if __name__ == "__main__":
	ptitle('Trying to generate windows grids on the test image...')

	path = 'test_images/test1.jpg'
	img = load_image(path)

	for config in WCONFIG['windows']:
	    boxes = slide_window(**config)
	    draw(draw_boxes(img, boxes))
