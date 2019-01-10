# Custom-class object detection

Train a neural net for custom class object detection and run inference at the edge.

## Contents



## Summary

This is a proposition for the EPFL Extension School Applied Machine Learning program Capstone project.


## Part 1 : building the data set

### Collecting data for Training

#### 1.1.	Collecting images from Google Image Search

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

##### To do :
    1. add more images for one class to perfect training
        adapt the code above to retrieve more than 100 images per search (limit set by Google)
    2. add other classes of objects

### Labelling and annotating train data

git clone https://github.com/Cartucho/OpenLabeling
–	creat bounding boxes
–	have outpu in xml format used by darkflow

### Collecting data for testing:

I attached a GoPro camera in my car and filmed my trips on Swiss highways. The footage captures many speedtraps. This will be used as a test set to evaluate the trained model acuracy.

![](test_set_gif_example.gif)


## Part 2 : Training the net

### Choosing the architecture of the net

For now, I chose to go with Single Shot Detector architectures, which are more likely to work on emnbedded devices to run inference at the edge (think Raspberry Pi 3 with limited computing power).

Yolo : quick prototyping

For compatibility purposes (I prototype on Mac OS X, then train on the cloud), I used this [`fork of Darknet`](https://github.com/thtrieu/darkflow) which is a Tensorflow implementation of Darknet.


##### To do
    1. try Tensorflow Object Detection API (https://github.com/tensorflow/models/tree/master/research/object_detection)
    2. consider implementing a SSD architecture from scratch in Tensorflow


### Training a Model using Google Colab GPU

#### Setting up Google Colab

1. Requirements : numpy, cython, opencv, Darkflow
2. using pre-trained weights : tiny-yolo-voc.weights
3. Modify cfg file to match number of classes to be trained with, in my case 1 :
    in the last layer : the number of classes should be changed from 80 to 1: classes=80 => classes=1
    in the convolutional layer above (the second to last layer) to num * (classes + 5) (30 in this case)
4. add a labels.text file with the labels of the classes to train with
5. upload the dataset and the annotations

#### Training

1. Start training for 1000 epochs

```
%cd /content/darkflow
!./flow --model cfg/tiny-yolo-voc-1c.cfg --train --dataset "/content/darkflow/input" --annotation "/content/darkflow/PASCAL_VOC" --gpu 1 --epoch 1000 --save 100
```

2. Assess results

Download the trained weights (in this case, checkpoint files), and run locally the net for inference on the test set.
After the first 1000 epochs, the model seems to have converged (at least loss was consistently below 0.8). But, the model did not predict any image on the test set. When run on the *train* set, bounding boxes appeared only when the thershold was lowered to 0,1 (at 0,5, maybe half the bounding boxes were showing).

3. Train for more epochs

After 1500 epochs

##### To do:
    1.	overfit the network on a very small dataset (a few images) before training on the whole dataset (the overfitting loss can be around or smaller than 0.1 for a one class only network). In the case of disabling noise augmentation, it can very well be near perfect 0.0.

##### Dataset
    1.	Need to have a better test set
    2.	other class to train with ? with more images for training and testing
    3.	train with more classes

## Part 3 : Running the trained net on a Raspberry Pi 3 for in-car inference

The last part of the project is on‑device inference with low latency and a small model size and *hopefully* decent fps.
