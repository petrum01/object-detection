###### Custom-class object detection

Train a neural net for custom class object detection and run inference at the edge.

### Contents



### Summary

This is a proposition for the EPFL Extension School Applied Machine Learning program Capstone project.


### Part 1 : building the data set

## Collecting data for Training

1.1.	Collecting images from Google Image Search

I used this [`repo on github`](https://github.com/hardikvasa/google-images-download) to collect images from the web.

I chose a custom class of objects (for now, speed-traps), and searched for specific keywords and also reverse searched for specific images, using the code below :

.. code-block:: python

    from google_images_download import google_images_download

    response = google_images_download.googleimagesdownload()   #class instantiation

    arguments = {

    #"similar_images":"https://www.scdb.info/blog/uploads/technology/11_gross.jpg",
    "keywords":"Traffic-Observer, radar autoroute suisse, schweizer autobahnblitzern, schweizer autobahnradar, speedtrap swiss",
    "output_directory":"new_model_data",
    "print_urls":True
    }
    paths = response.download(arguments)   #passing the arguments to the function
    print(paths)   #printing absolute paths of the downloaded images

Then renamed them and placed them in an appropriate folder structure:

.. code-block:: python

    import os

    imdir = 'images'
    if not os.path.isdir(imdir):
        os.mkdir(imdir)

        radar_folders = [folder for folder in os.listdir('new_model_data_copie') if 'radar' in folder]
        #radar_folders = os.listdir('new_model_data_copie')
        print(radar_folders)

        n = 0
        for folder in radar_folders:
        for imfile in os.scandir(os.path.join('/Users/pm/Documents/AI/compvision/clean/new_model_data_copie/',folder)):
            os.rename(imfile.path, os.path.join(imdir, '{:06}.png'.format(n)))
            n += 1


To do :
    1. add more images for one class to perfect training
        adapt the code above to retrieve more than 100 images per search (limit set by Google)
    2. add other classes of objects


Collecting data for testing:

I attached a GoPro camera in my car and filmed my trips on Swiss highways. The footage captures many speedtraps. This will be used as a test set to evaluate the trained model acuracy.

    Example :
![](test_set_gif_example.gif)


### Part 2 : Training the net

## Choosing the architecture of the net

For now, I chose to go with Single Shot Detector architectures, which are more likely to work on emnbedded devices to run inference at the edge (think Raspberry Pi 3 with limited computing power).

Yolo : quick prototyping

For compatibility purposes (I prototype on Mac OS X, then train on the cloud), I used this [`fork of Darknet`](https://github.com/thtrieu/darkflow) which is a Tensorflow implementation of Darknet.


# To do
    1. try [`Tensorflow Object Detection API`](https://github.com/tensorflow/models/tree/master/research/object_detection)
    2. consider implementing a SSD architecture from scratch in Tensorflow


## Training a Model
