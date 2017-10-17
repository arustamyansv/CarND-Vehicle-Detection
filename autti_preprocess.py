import cv2
from glob import glob
import numpy as np
import random
import csv
import os.path

dspath = 'data/object-dataset'
f = open('{}/labels.csv'.format(dspath), newline='')
reader = csv.reader(f, delimiter=' ', quotechar='"')

for row in reader:
    # parse columns
    frame = row[0]
    xmin, ymin = int(row[1]), int(row[2])
    xmax, ymax = int(row[3]), int(row[4])
    occluded = int(row[5])
    label = row[6]
    
    # no trucks needed 
    if label == 'truck':
        continue
    
    # split it by directories so we will not preprocess it later
    ipath = 'data'
    if label == 'car':
        ipath = '{}/vehicles/autti'.format(ipath)
        
        if occluded == 1:
            ipath = '{}_occluded'.format(ipath)
    else:
        ipath = '{}/non-vehicles/autti'.format(ipath)
    
    ipath = '{}/{}_{}_{}_{}_{}'.format(ipath, xmin, ymin, xmax, ymax, frame)
    
    if os.path.isfile(ipath):
        print ('Skipped: {}'.format(ipath))
    
    image = cv2.imread('{}/{}'.format(dspath, frame))
    image = image[ymin:ymax,xmin:xmax,:]
    image = cv2.resize(image, (64, 64))
        
    cv2.imwrite(ipath, image)
    print ('Image processed: {}'.format(ipath))
    
f.close()
