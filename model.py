from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from progressbar import ProgressBar
# import matplotlib.pyplot as plt
import numpy as np
import random

from utils import *

MCONFIG = {
    'vehicles_paths': [
        'data/vehicles/GTI_Far',
        'data/vehicles/GTI_Left',
        'data/vehicles/GTI_MiddleClose',
        'data/vehicles/GTI_Right',
        'data/vehicles/KITTI_extracted',
        # 'data/vehicles/autti',
        # 'data/vehicles/autti_occluded'
    ],
    'non-vehicles_paths': [
        'data/non-vehicles/Extras',
        'data/non-vehicles/GTI',
        # 'data/non-vehicles/autti'
    ],

    'split_train_size': 0.2,

    'model_file': 'model.pkl'
}

if __name__ == "__main__":
	ptitle('Loading image paths from folders...')

	vehicles = load_images(MCONFIG['vehicles_paths'])
	print ('Vehicles: {}'.format(len(vehicles)))

	non_vehicles = load_images(MCONFIG['non-vehicles_paths'])
	print ('Non-vehicles: {}'.format(len(non_vehicles)))

	samples = np.array(vehicles + non_vehicles)


	ptitle('Prepare labels and features...')

	# extract all features
	features = []
	bar = ProgressBar()
	for i in bar(range(len(samples))):
		img = load_image(samples[i])
		features.append(extract_features(img))

	X = np.array(features, dtype='float64')
	y = np.hstack((np.ones(len(vehicles)), np.zeros(len(non_vehicles))))

	print ('Features shape: {}'.format(X.shape))

	ptitle('Scaling features...')

	# scaler = RobustScaler()
	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)

	ptitle('Training process...')

	print ('Splitting features: 80% train set, 20% test set')
	rand = np.random.randint(0, 100)
	data = train_test_split(X, y, test_size=MCONFIG['split_train_size'], random_state=rand)
	X_train, X_test, y_train, y_test = data

	clf = LinearSVC(dual=False)

	print ('Feeding classifier with data...')
	clf.fit(X_train, y_train)

	print ('Score: {}'.format(clf.score(X_test, y_test)))

	ptitle('Model saving...')

	joblib.dump([clf, scaler], MCONFIG['model_file'])

	print ('Model saved: {}'.format(MCONFIG['model_file']))
	print ('Complete!!')
