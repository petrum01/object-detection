# Custom-class object detection

Train a neural net for custom class object detection and run inference at the edge.

## Contents
<!-- TOC depthFrom:1 depthTo:4 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Custom-class object detection](#custom-class-object-detection)
	- [Contents](#contents)
	- [Summary](#summary)
	- [Part 1 : building the data set](#part-1-building-the-data-set)
		- [Collecting data for Training](#collecting-data-for-training)
			- [Collecting images from Google Image Search](#collecting-images-from-google-image-search)
			- [Collecting images from Google street view](#collecting-images-from-google-street-view)
			- [Collecting images from live GoPro footage](#collecting-images-from-live-gopro-footage)
		- [Labelling and annotating train data](#labelling-and-annotating-train-data)
		- [Collecting data for testing:](#collecting-data-for-testing)
	- [Part 2 : Training the net](#part-2-training-the-net)
		- [Choosing the architecture of the net](#choosing-the-architecture-of-the-net)
		- [Training the model](#training-the-model)
			- [Training the model using Google Colab GPU](#training-the-model-using-google-colab-gpu)
	- [Part 3 : Running the trained model on a mobile device for in-car inference](#part-3-running-the-trained-model-on-a-mobile-device-for-in-car-inference)

<!-- /TOC -->

## Summary

This is a proposition for the Capstone project for the EPFL Extension School Applied Machine Learning program. The objective is to train a neural net for custom class object detection and run inference at the edge by:
- building a custom data set and annotate it;
- train a network using data augmentation techniques and transfer learning with fine-tuning of the last layers;
- running inference at the edge on a device with limited computing power.

I will thoroughly document each phase of the project and draw conclusions on the best techniques to use.

## Part 1 : building the data set

### Collecting data for Training

#### Collecting images from Google Image Search

I used this [`repo on github`](https://github.com/hardikvasa/google-images-download) to collect images from the web.

I chose a custom class of objects (for now, speed-traps), and searched for specific keywords and also reverse searched for specific images, using the code below :

```python
from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {
"keywords":"Traffic-Observer, radar autoroute suisse, schweizer autobahnblitzern, schweizer autobahnradar, speedtrap swiss",
"similar_images":"https://www.scdb.info/blog/uploads/technology/11_gross.jpg",
"output_directory":"new_model_data",
"print_urls":True
}
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images
```

Then renamed them and placed them in an appropriate folder structure:

```python

import os

imdir = 'images'
if not os.path.isdir(imdir):
    os.mkdir(imdir)

    radar_folders = [folder for folder in os.listdir('new_model_data') if 'radar' in folder]
    print(radar_folders)

    n = 0
    for folder in radar_folders:
    for imfile in os.scandir(os.path.join('/Users/pm/Documents/AI/compvision/clean/new_model_data_copie/',folder)):
        os.rename(imfile.path, os.path.join(imdir, '{:06}.png'.format(n)))
        n += 1
```

#### Collecting images from Google street view


#### Collecting images from GoPro footage

important to have a train dataset close to what the test set might be (how to say this ??)

Methodology for filming :
- filming at 60 fps, 720p super wide mode, with GoPro Hero+
- camera mounted on the dashboard of the car
- filming the same object with different lighting conditions, slightly different angles

Post-processing in iMovie
simple color grading


Extracting frames from video footage
run $ python extract_frames.py


200 to 1000 images


Issues :
how many frames to take ?
what about the similiraty of frames that are close to each other ?
point above has an impact on train / test set

- create a balanced dataset (with and without radar, repartition to match 'real' distribution in order to keep false positive ratio low)

then, use keras image processing for data augmentation
ImageDataGenerator for real-time data augmentation

### Labelling and annotating train data

git clone https://github.com/Cartucho/OpenLabeling
- create bounding boxes
- generate output in xml format used by darkflow

### Collecting data for testing:

I attached a GoPro camera in my car and filmed my trips on Swiss highways. The footage captures many speedtraps. This will be used as a test set to evaluate the trained model acuracy.

![](test_set_gif_example.gif)


## Part 2 : Training the net

### Choosing the architecture of the net

In the field of computer vision, many pre-trained models are now publicly available, and can be used to bootstrap powerful vision models out of very little data.

For now, I chose to go with Single Shot Detector architectures, which are more likely to work on emnbedded devices to run inference at the edge (think Raspberry Pi 3 with limited computing power).

For compatibility purposes (I prototype on Mac OS X, then train on the cloud), I used this [`fork of Darknet`](https://github.com/thtrieu/darkflow) which is a Tensorflow implementation of Darknet.


##### To do
    1. try Tensorflow Object Detection API (https://github.com/tensorflow/models/tree/master/research/object_detection)
    2. consider implementing a SSD architecture from scratch in Tensorflow
	3. Tensorflow Lite

### Training the model

Transfer learning and fine-tuning of the last layers sing data augmentation techniques with Keras ImageDataGenerator.

#### Training the model using Google Colab GPU

##### Setting up Google Colab

1. Requirements : numpy, cython, opencv, Darkflow
2. using pre-trained weights : tiny-yolo-voc.weights
3. Modify cfg file to match number of classes to be trained with, in my case 1 :
    in the last layer : the number of classes should be changed from 80 to 1: classes=80 => classes=1
    in the convolutional layer above (the second to last layer) to num * (classes + 5) (30 in this case)
4. add a labels.text file with the labels of the classes to train with
5. upload the dataset and the annotations

##### Training

1. Start training for 1000 epochs

```
%cd /content/darkflow
!./flow --model cfg/tiny-yolo-voc-1c.cfg --train --dataset "/content/darkflow/input" --annotation "/content/darkflow/PASCAL_VOC" --gpu 1 --epoch 1000 --save 100
```

2. Assess results

Download the trained weights (in this case, checkpoint files), and run locally the net for inference on the test set.
After the first 1000 epochs, the model seems to have converged (at least loss was consistently below 0.8). But, the model did not predict any image on the test set. When run on the *train* set, bounding boxes appeared only when the thershold was lowered to 0,1 (at 0,5, maybe half the bounding boxes were showing).

3. Train for more epochs

After 1500 epochs, same

##### To do:
    1.	overfit the network on a very small dataset (a few images) before training on the whole dataset (the overfitting loss can be around or smaller than 0.1 for a one class only network). In the case of disabling noise augmentation, it can very well be near perfect 0.0.

Trained on a small dataset, 3 images, with 4 objects in them of the same class. The model converged after approx. 5000 epochs.
Saved weights after 8000 epochs in /Users/pm/Documents/AI/compvision/darkflow_master/darkflow/TRAINING/overfit_on_small_data_set/ckpt


## Part 3 : Running the trained model on a mobile device for in-car inference

The last part of the project is on‑device inference with low latency and a small model size and *hopefully* decent fps.

Running inference on compute-heavy machine learning models on mobile devices is resource demanding due to the devices’ limited processing and power

Hardware
- Raspberry Pi 3
- Raspberry Pi 3 w/ GPU card
- iPhone or Android phone
