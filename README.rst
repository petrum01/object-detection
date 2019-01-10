Custom-class object detection
#############################

Train a neural net for custom class object detection and run inference at the edge.

Contents

.. contents:: :local:

Summary
=======

This is a proposition for the EPFL Extension School Applied Machiune Learning program Capstone project.


Part 1 : building the data set
==============================

## Collecting data for Training

1.1.	Collecting images from Google Image Search

I used this `repo on github <https://github.com/hardikvasa/google-images-download>` to collect images from the web.

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

Then renamed them:

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
1. add more images for one class
2. add other classes of objects
3.




1.	but only 150 imgs,
2.	choose another class if needed
3.	solve issue with limitations to 100 imgs




## Training




#### Part 2

First portable device to improve users’ safety by using deep learning to do live object detection and reporting to center of management
