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
			- [Collecting images from GoPro footage](#collecting-images-from-gopro-footage)
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

#### First try : Collecting images from Google Image Search

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

It resulted in a very limited dataset (about 100 images), not coherent enough :different types of speed traps, different POV.


#### Second try : Collecting images from GoPro footage

I attached a GoPro camera in my car and filmed my trips on Swiss highways. The footage captures many speedtraps. This will be used as a train / test set.

![](test_set_gif_example.gif)

Methodology for filming :
- filming at 60 fps, with GoPro Hero+
- camera mounted on the dashboard of the car
- filming the same object with different lighting conditions, slightly different angles

Post-processing in iMovie:
- simple color grading

Extracting frames from video footage
run $ python extract_frames.py

The final train set consists of :
- 400 images
- one class of object (speedtrap)

This dataset is derived from footage that, for each unique object, is filmed by a dash-mounted camera:
- from right and left lane
- from different camera position / angle
- in the two ways
- in at least two different lighting conditions

Issues :
- what about the similiraty of frames that are close to each other ?


### Labelling and annotating train data

In RectLabel:
- create bounding boxes for each image
- generate annotation output in xml format

Quick check of the output vs. the original images :

```python
import os
annot_dir = '/Users/pm/Documents/AI/compvision/Object_detection/creating_dataset/GP020472_edit_close_annot'
img_dir = '/Users/pm/Documents/AI/compvision/Object_detection/creating_dataset/GP020472_edit_close_img'
filesA = [os.path.splitext(filename)[0] for filename in os.listdir(annot_dir)]
filesB = [os.path.splitext(filename)[0] for filename in os.listdir(img_dir)]
print ("images without annotations:",set(filesB)-set(filesA))
```

- Folder structure:
object-detection/
	GP020472_edit_close_img/ (refers to the mp4 file used to extract images from)
		images.jpg & annotations.xml
		train/
			copy of img.jpg and annot.xml used for training
			test/
			copy of img.jpg and annot.xml used for testing
	data/
	training/


In each train & test folders, copy images and annotation xml files.

To use a custom dataset in Tensorflow Object Detection API, you must convert it into the TFRecord file format:
- Convert XML files to singular CSV files : xml_to_csv.py


xml files that can be then converted to the TFRecord files.
- rectlabel_create_pascal_tf_record.py file to be copied in /Users/pm/Documents/AI/compvision/Object_detection/tfod/models/research/object_detection/dataset_tools
- copy annotations in images folder (images/annotation/)
- run : (once for train, once for test)
$ python3 rectlabel_create_pascal_tf_record.py --images_dir="/Users/pm/Documents/AI/compvision/Object_detection/creating_dataset/GP020472_edit_close_img" --image_list_path="/Users/pm/Documents/AI/compvision/Object_detection/creating_dataset/test.txt" --label_map_path="/Users/pm/Documents/AI/compvision/Object_detection/creating_dataset/label_map.pbtxt" --output_path="/Users/pm/Documents/AI/compvision/Object_detection/creating_dataset/test.record"

source : https://rectlabel.com/help#tf_record


## Part 2 : Training the net

### Choosing the architecture of the net

In the field of computer vision, many pre-trained models are now publicly available, and can be used to bootstrap powerful vision models out of very little data.

For now, I chose to go with Single Shot Detector architectures, which are more likely to work on emnbedded devices to run inference at the edge (think Raspberry Pi 3 with limited computing power).


### First try : Training the model on Darkflow / YOLO

For compatibility purposes (I prototype on Mac OS X, then train on the cloud), I used this [`fork of Darknet`](https://github.com/thtrieu/darkflow) which is a Tensorflow implementation of Darknet.

Transfer learning reduces the training time and data needed to achieve a custom task. It takes a CNN that has been pre-trained, removes the last fully-connected layer and replaces it with the custom fully-connected layer, treating the original CNN as a feature extractor for the new dataset, and the last fully-connected layer - after training - acts as the classifier for the new dataset.


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
After several tries (1000 epochs to 8000 epochs), the model seems to have converged (at least loss was consistently below 0.8). But, the model did not predict any image on the test set. When run on the *train* set, bounding boxes appeared only when the thershold was lowered to 0,1 (at 0,5, maybe half the bounding boxes were showing).


### Second try : Training the model using Tensorflow Object Detection API


Please refer to Training on Paperspace.md


## Part 3 : Running the trained model on a mobile device for in-car inference

The last part of the project is on‑device inference with low latency and a small model size and *hopefully* decent fps.

Running inference on compute-heavy machine learning models on mobile devices is resource demanding due to the devices’ limited processing and power

Hardware
- Raspberry Pi 3
- Raspberry Pi 3 w/ GPU card
- iPhone or Android phone
- Nvidia Jetson
