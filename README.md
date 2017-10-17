**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1]: ./output_images/car.png
[image2]: ./output_images/not_car.png
[image3]: ./output_images/hist.png
[image4]: ./output_images/bin_spatial.png
[image5]: ./output_images/hog1.png
[image6]: ./output_images/hog2.png
[image7]: ./output_images/hog3.png

[image8]: ./output_images/window1.png
[image9]: ./output_images/window2.png
[image10]: ./output_images/window3.png

[image11]: ./output_images/pred-1.png
[image12]: ./output_images/pred-2.png
[image13]: ./output_images/pred-3.png
[image14]: ./output_images/pred-4.png
[image15]: ./output_images/pred-5.png

[image16]: ./output_images/result-1.png
[image17]: ./output_images/result-2.png
[image18]: ./output_images/result-3.png
[image19]: ./output_images/result-4.png
[image20]: ./output_images/result-5.png

[video1]: ./project_video_complete.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Images Processing 

#### 1. Images Augmentation
The whole process starts from `model.py` file. This file contains data loading and model training.
I first train classifier only using main dataset. However later on I tried and add Autti dataset to model. Code for Autti dataset extraction is in the file `autti_preprocess.py`. It run extraction of objects from autti images and rescaling them to 64x64 in advance. Taking everything except tracks.

Here are random examples from final dataset:

Vehicle:

![alt text][image1]

Non-vehicle:

![alt text][image2]

#### 2. Features Extraction

##### 1. HOG & parameters explanation
Code for step is contained in file  `utils.py`. There are several methods for HOG features. Main one is `extract_features_hog`. I also utilised HOG subsampling. Method for subsampling + windows scaling iss defined in `utils.py` also (`extract_features_hog_scaled`).
I tried different options for HOG extraction trying to change color space or number of color layers from each color space plus meta option of `skimage.hog()` method itself. 

The main criteria was to get the most heterogeneity from the image features so classifier will have something to work with.
Two major changes was switching to `YCrCb` colorspace and increasing number of layers utilised as features to all layers. After those changes different values made unsignificant accuracy changes.

However best variant was:

`Color space: YCrCb`

`Orientations: 9`

`Pixels Per Cell: (8, 8)`

`Cells Per Block: 2`

I grabbed random image from car class and displayed it to get a feel for what the `skimage.hog()` output looks like.
Here is an example using options above:

![alt text][image5]

![alt text][image6]

![alt text][image7]

##### 2. Other feature extractions
In addition to HOG I implemented bin spatial and histogram features extraction. Code for this steps also contained in file `utils.py` in methods `extract_features_spatial` and `extract_features_hist` respectively.
Feature list became pretty big but it showed better accuracy so I decide to deal with it...

Spatial size: (32, 32)

Spatial features:

![alt text][image4]

Histogram Bins: (32)

Here is histogram map:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained linear SVC using file `model.py`. 
In order to do this i first split by dataset 80% for train and 20% for test dataset. Then run model training and score evaluation for the test set.

As a result I got pretty reasonable accuracy of 98.8%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

Code for this block is contained in file `windows.py` file. Method takes configuration from WCONFIG object and iteratively create grids relevant to configuration specified. Method is pretty simple and approach was based on udacity example  i need to count number of possible windows in the secified area and then generate windowses itself.
However I changed suggested approach with 'overlay' to shifts in pixels. It looks clearer on my mind. 

I choose grid side approximately and based on recommendations in the community.

Here are my windows grids:

![alt text][image8]

![alt text][image9]

![alt text][image10]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a result. Here are examples of labels map and classified image:

I'll put several images with predicted boxes and composed final result acording to heat map below.

Estimated Boxes:

![alt text][image11]

![alt text][image12]

![alt text][image13]

![alt text][image14]

![alt text][image15]


Final Result:

![alt text][image16]

![alt text][image17]

![alt text][image18]

![alt text][image19]

![alt text][image20]


As for optimisation I implemented sub-sampling windows search and this significantly increase speed of generation. I also tried to keep as less data in the memory as I could, unfortunately I didn't find a way to pass generator to feed this model.


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Final video provided in the submission archive in the file `project_video_complete.mp4`.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. Code for this part could be found in `detect_vehicle.py` file. Heatmap processing separated in the `compose_heatmap` function. 
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
This part included in the method `detect` as a final part of detection pipeline.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

During the implementation I faced a big issue which is resources and time needed for big datasets. There was no possibility to process full autti dataset at all. Probably the solution will be to split dataset among numerous classifiers and apply list of classifiers during image processing. As a bonus it will allow to split teaching process among separate CPU cores.

My pipeline will fail for sure in case somebody will drive to the left of my car. I choose small horisontal size in order to increase a processing speed so in order to work in this case configuration must be changed.
I assume it can faile cause of bad weather conditions, bigger hills etc.
I also think it's possible to make heatmap and box processing more advanced. 
