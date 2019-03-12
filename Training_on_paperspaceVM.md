
## First, let's set up the environnement

mkdr tfod
cd tfod
git clone https://github.com/tensorflow/models
virtualenv tfodenv
source activate tfodenv

pip install tensorflow
Cython - 0.29.3
contextlib2 - 0.5.5
pillow - 5.4.1
lxml - 4.3.0
jupyter
matplotlib

cd models/research
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.

/*

nano .bashrc
add this line :
export PYTHONPATH=$PYTHONPATH:`/home/paperspace/tfod/models/research`:`/home/paperspace/tfod/models/research`/slim

shutdown machine and restart
sudo shutdown -P now

python object_detection/builders/model_builder_test.py

git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
make install
python setup.py install


## Configuring the training pipeline

I will use one of the pre-trained models provided in TensorFlow Object detection API.
The model we will be using in this case is ssd_mobilenet_v1_coco. It is always a trade-off between accuracy and speed.

Download :
- configuration file : https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config
- the latest pre-trained weights for the model : http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz , extract its content in training_demo/pre-trained-model/

Below are the changes that we shall need to apply to the downloaded .config file:

```
model {
  ssd {
    num_classes: 1  #number of classes to be trained on

[...]

train_config: {
  batch_size: 24 #increase or decrease depending of GPU memory usage

[...]

  fine_tune_checkpoint: "/home/paperspace/tensorflow/training_PM/pre-trained-model/model.ckpt"    #Path to extracted files of pre-trained model

  from_detection_checkpoint: true
[...]

train_input_reader: {
  tf_record_input_reader {
    input_path: "/home/paperspace/tensorflow/training_PM/data/train.record"   #Path to training TFRecord file
  }
  label_map_path: "/home/paperspace/tensorflow/training_PM/data/label_map.pbtxt"  #Path to label map file
}

[...]

eval_input_reader: {
  tf_record_input_reader {
    input_path: "/home/paperspace/tensorflow/training_PM/data/test.record"  # Path to testing TFRecord
  }
  label_map_path: "/home/paperspace/tensorflow/training_PM/data/label_map.pbtxt"   # Path to label map file

}
```
## Upload data set and config files to Paperspace VM

ssh to the machiune and :
scp -r /localfolder/ paperspace@[publicIP]:./Desktop/


## Training the model

Before we begin training our model, let’s go and copy the TensorFlow/models/research/object_detection/legacy/train.py script and paste it straight into our training_demo folder. We will need this script in order to train our model.

Now, to initiate a new training job, cd inside the training_demo folder and type the following:

$ python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config

### raises error :
## ValueError: Unknown ssd feature_extractor: ssd_mobilenet_v1_coco

changed :     feature_extractor {
      type: 'ssd_mobilenet_v1_coco'

to :      feature_extractor {
      type: 'ssd_mobilenet_v1'

## tensorflow.python.framework.errors_impl.NotFoundError: ~/tfod/workspace/training_demo/training/label_map.pbtxt; No such file or directory
replacing relative path with absolute path in config


## monitoring with tensorboard

tensorboard --logdir=training\

NO SOLUTION FOUND YET

## Exporting a Trained Inference Graph

Copy the TensorFlow/models/research/object_detection/export_inference_graph.py script and paste it straight into your training_demo folder.

/Users/pm/Documents/AI/compvision/Object_detection/creating_dataset/Object-Detection/training_demo/export_inference_graph.py

scp /Users/pm/Documents/AI/compvision/Object_detection/creating_dataset/Object-Detection/training_demo/export_inference_graph.py paperspace@184.105.136.175:/home/paperspace/tfod/workspace/training_demo

Now, cd inside your training_demo folder, and run the following command, specifiying the checkpoint file with the largest step.

python export_inference_graph.py --input_type image_tensor --pipeline_config_path pipeline_PM.config --trained_checkpoint_prefix training/model.ckpt-9196 --output_directory trained-inference-graphs/output_inference_graph_PM_fpn

download graph on local machine
scp -r paperspace@184.105.117.103:/home/paperspace/tensorflow/training_PM_fpn/trained-inference-graphs/output_inference_graph_PM_fpn /Users/pm/Downloads


# Testing the fine-tuned model

After training for approx. 1 hour on remote server (Paperspace),

copy output_inference_graph_v1.pb folder (dowloaded from paperspace) to models/research/object_detection
copy label_map.pbtxt to models/research/object_detection/data

Now, we're just going to use the sample notebook, edit it, and see how our model does on some testing images. I copied some of my models/object_detection/images/test images into the models/object_detection/test_images directory, and renamed them to be image3.jpg, image4.jpg...etc.

object_detection_tutorial_PM2.py (from models/research/object_detection) is working in visionclean environnement.
save object_detection_tutorial.ipynb as :
object_detection_tutorial_inf_test.py

A few changes. First, head to the Variables section, and let's change the model name, and the paths to the checkpoint and the labels:

# What model to download.
MODEL_NAME = 'output_inference_graph_v1.pb'  

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'label_map.pbtxt')
NUM_CLASSES = 1

Next, we can just delete the entire Download Model section, since we don't need to download anymore.

Finally, in the Detection section, change the TEST_IMAGE_PATHS var to:

TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 8) ]

With that, you can go to the Cell menu option, and then "Run All."


# errors :
ValueError: NodeDef mentions attr 'Truncate' not in Op<name=Cast; signature=x:SrcT -> y:DstT; attr=SrcT:type; attr=DstT:type>; NodeDef: ToFloat = Cast[DstT=DT_FLOAT, SrcT=DT_UINT8, Truncate=false](image_tensor). (Check whether your GraphDef-interpreting binary is up to date with your GraphDef-generating binary.).

update TF:
1.12 on papaerspace
1.10 on conda env

steps:
conda remove -n visionclean tensorflow
conda remove -n visionclean tensorboard
conda install -c anaconda tensorflow

on conda forge, it's tf 1.10 !
on anaconda it is 1.12

# error:

OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.

to solve it :
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# output after checkpoint 270 (approx 1 hour training on paperspace VM):
works on jupyter notebook :
object_detection_tutorial-inf_test.ipynb

but does not detect any test images (one false positive in sample img)

# output after checkpoint 1000 :
output_inference_graph_v1_2.pb
no Detection

adding 4 images from train set : image8.jpg to image 11.jpg
no Detection



# sources
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/
also :
https://pythonprogramming.net/testing-custom-object-detector-tensorflow-object-detection-api-tutorial/


## Fine-tuning a model using model_main.py instead of train.py
source : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md#running-the-training-job

BEFORE : just try with main_model
copy model_main.py to /home/paperspace/tfod/workspace/training_demo

# From the tfod/workspace/training_demo directory
PIPELINE_CONFIG_PATH=training/ssd_mobilenet_v1_coco.config
MODEL_DIR=training/
NUM_TRAIN_STEPS=150000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr


##Error :
ModuleNotFoundError: No module named 'pycocotools'

##SOLUTION
git clone https://github.com/pdollar/coco.git

cd coco/PythonAPI
make
make install
python setup.py install

## error
tensorboard not accessible from my local machine

# export graph and downlad it
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix training/model.ckpt-331 --output_directory trained-inference-graphs/output_inference_graph_v1_3.pb

download graph on local machine
scp -r paperspace@184.105.136.175:/home/paperspace/tfod/workspace/training_demo/trained-inference-graphs/output_inference_graph_v1_3.pb /Users/pm/Downloads

# monitor with tensorboard:
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
unzip ngrok-stable-linux-amd64.zip

launch tensorboard
tensorboard --logdir=training
./ngrok http 6006

(source : https://ngrok.com/docs)

# monitor GPU usage
$ nvidia-smi -l


# Error: GPU ?
the training process (using model_main.py) uses 400% or more of the CPU (check with $top) and no GPU
# Solution :
change the batch size to 1
No effect
# Solution : print available devices seen by TF
$ python3 -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"

2019-02-24 08:24:53.604765: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 13820102785163268728
, name: "/device:XLA_CPU:0"
device_type: "XLA_CPU"
memory_limit: 17179869184
locality {
}
incarnation: 3064048576877597110
physical_device_desc: "device: XLA_CPU device"
]
# Error : tensorflow does not recognize the GPU
tf.test.gpu_device_name()
tf.test.is_gpu_available()

$ python3 -c "import tensorflow as tf; print(tf.test.gpu_device_name()); print(tf.test.is_gpu_available())"

2019-02-24 08:33:27.155876: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA

False
#Solution : reinstall tensorflow-gpu
$ pip uninstall tensorflow
$ pip install tensorflow-gpu
not working, CUDA drivers are 9.1, and should ne 9.0 ??


#Solution : install tensorflow-gpu on fastai env
no

# solution : Tensorflow GPU requirements :
NVIDIA® GPU card with CUDA® Compute Capability 3.5 or higher : OK, Quadro P4000 6.1

The question is CUDA 9.0 vs 9.1. From what I have read it seems that version 9.1 works if building from source, whereas 9.0 works out of the box when installing from the binaries.


# NEW Machine
cat /usr/local/cuda/version.txt
and
nvcc --version
are the same

and tensorfloww detects gpu
$ python3 -c "import tensorflow as tf; print(tf.test.is_gpu_available())"

# Train on new Machine
## First, let's set up the environnement


cd Projects
git clone https://github.com/tensorflow/models
virtualenv tfodenv         #no need for virtualenv
source activate tfodenv    #no need for virtualenv

sudo apt install python-pip

pip install tensorflow     #already installed, version

tensorboard (1.11.0)
tensorflow (1.11.0rc1)
tensorflow-gpu (1.10.1)

Cython - 0.29.3   #Cython (0.28.5) already installed,
contextlib2 - 0.5.5
pillow - 5.4.1  #Pillow (5.1.0)
lxml - 4.3.0  #no
jupyter
matplotlib

cd models/research
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.

/*

nano .bashrc
add this line :
export PYTHONPATH=$PYTHONPATH:/home/paperspace/tensorflow/models/research:/home/paperspace/tensorflow/models/research/slim


shutdown machine and restart
sudo shutdown -P now

python object_detection/builders/model_builder_test.py

git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
make install
python setup.py install
cp -r pycocotools /home/paperspace/tensorflow/models/research/

#New folder architecture
home/paperspace/tensorflow/
├─ models
│ ├─ official
│ ├─ research
│ ├─ samples
│ └─ tutorials
└─ training_PM
    └─ data
        ├─ label_map.pbtxt
        ├─ test.record
        └─ train.record
    └─ pre-trained-model
        └─ ssd_mobilenet_v1_coco (where the pre-trained model checkpoints are)

#copy model_main.py to training_PM
# From training_PM directory
PIPELINE_CONFIG_PATH=pipeline_PM.config
MODEL_DIR=training/
NUM_TRAIN_STEPS=25000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr

error :
Object was never used (type <class 'tensorflow.python.framework.ops.Tensor'>):
<tf.Tensor 'report_uninitialized_variables_1/boolean_mask/GatherV2:0' shape=(?,) dtype=string>
If you want to mark it as used call its "mark_used()" method.

pip uninstall tensorflow (1.11.0rc1)
pip install tensorflow

# monitor with tensorboard:
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
unzip ngrok-stable-linux-amd64.zip

launch tensorboard
tensorboard --logdir=training
./ngrok http 6006

(source : https://ngrok.com/docs)

#STILL running on CPU only !
updated tf-gpu...
uninstalled tf
ImportError: libcublas.so.9.0: cannot open shared object file: No such file or directory
adding to bashrc :
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

ERROR : ImportError: libcublas.so.9.0: cannot open shared object file: No such file or directory
so it is looking for CUDA 9.0

Install CUDA 9.0
https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73

### works on GPU with CUDA 9.0 ###

## Hyperparameters tuning :

- training_PM_1:
'default settings' of config file
11k steps
batch_size = 24

?? ATTENTION :
 did I export the frozen graph correctly ? (did I reference MY config file ??)

- training_PM_2 :
26k steps
batch_size = 1
detects eveything at first...and then nothing !
mAP close to zero

?? ATTENTION :
 did I export the frozen graph correctly ? (did I reference MY config file ??)


- training_PM_2 :
batch_size = 1024
GPU memory usage is saturated but volatge is low
trains very slowly

- training_PM_3 :
batch_size = 192
ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[192,256,38,38] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc

batch_size = 48

?? ATTENTION :
 did I export the frozen graph correctly ? (did I reference MY config file ??)

- training_PM_4 :
12k steps
batch_size = 48
mAP 0,32
mAP @.50IOU 0,72
loss 1.3
exported inference graph to /Users/pm/Documents/AI/compvision/Object_detection/tfod/models/research/object_detection/output_inference_graph_PM_4

slower to train due to data augment options (it seems the data augmentation operations are done by CPU before each batch/step and then processed by GPU, many I/O)
data_augmentation_options {
  random_horizontal_flip {
  ssd_random_crop {
  random_adjust_brightness {
  random_adjust_contrast {
  random_adjust_hue {
  random_adjust_saturation {
  random_distort_color {

source : https://github.com/tensorflow/models/blob/master/research/object_detection/core/preprocessor.py

- training_PM_5:
redo with the 'default settings' of config file
batch_size = 24
after 20k steps:
mAP 0,37
mAP @.50IOU 0,76
loss <1
exported frozen graph : output_inference_graph_PM_5

- training_PM_5b:
continuing training_PM_5 to more steps

To re-start training a model already trained for xx steps, do not change in config file :
fine_tune_checkpoint (set to: pre-trained-model/ssd_mobilenet_v1_coco/model.ckpt)
Specifying the "training" folder when running model_main.py is enough and restores parameters from latest checkpoint (see message below).
OK : INFO:tensorflow:Restoring parameters from training/model.ckpt-20000




- training_PM_fpn

SSD with Mobilenet v1 FPN feature extractor, shared box predictor and focal loss (a.k.a Retinanet). (See Lin et al, https://arxiv.org/abs/1708.02002)
Trained on COCO, initialized from Imagenet classification checkpoint
Achieves 29.7 mAP on COCO14 minival dataset.

/Users/pm/Documents/AI/compvision/Object_detection/creating_dataset/Object-Detection/training_PM_fpn

Download :
ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config
from : https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config

Download latest pre-trained weights for the model :
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz

with default config params (batch_size : 64) : OOM
with batch_size : 32  // OOM
24

- in config file :
image_resizer {
  fixed_shape_resizer {
    height: 300  #down from 640
    width: 300  #down from 640
  }
ERROR : ValueError: Dimensions must be equal, but are 20 and 19 for 'FeatureExtractor/MobilenetV1/fpn/top_down/add' (op: 'Add') with input shapes: [24,20,20,256], [24,19,19,256].

SOlution : in config add "pad_to_multiple: 1" in the feature_extractor section (after depth_multiplier: 1.0)
same error

 - w/ original config (640*640), just lowering batch size, starting to 1

trains, but loss is very high and :
ERORR type :
WARNING:root:Variable [MobilenetV1/Conv2d_0/BatchNorm/beta] is not available in checkpoint
solution: added "from_detection_checkpoint: true" in config file

and also duplicates logging
solution : Open variables_helper.py in models/research/object_detection/utils/variables_helper.py and replace all occurrences of logging with tf.logging

OK, trains Now
loss is huge

trying batch_size : 8
steps : 9196
mAP:0.70
mAP 0.50 : 0.99
loss:<1.50





- training_PM_resnet

2 types of ERROR :
- WARNING:root:Variable [resnet_v1_50/conv1/BatchNorm/gamma] is not available in checkpoint
and
- ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[16,1024,40,40] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc

tried reducing batch size but not working






# About batch size
Number of parameters of the model

available GPU memory bytes  = 7491MiB
one mebibyte is equal to 1048576 bytes (https://en.wikipedia.org/wiki/Mebibyte)

number of trainable parameters = 5,1 M

Max batch size= available GPU memory bytes / 4 / (size of tensors + trainable parameters)
source : https://stackoverflow.com/questions/46654424/how-to-calculate-optimal-batch-size

Anyway, it seems to crash (OOM error) for batch_size starting at 192 (maybe even lower).

# mAP
https://stackoverflow.com/questions/47692742/tensorflow-object-detection-api-evaluation-map-behaves-weirdly

![graph for mean average precision at 0.5 IOU (mAP@.50IOU)](training_PM_3_mAP.png)
Mean average precision measures our model’s percentage of correct predictions for all labels. IoU is specific to object detection models and stands for Intersection-over-Union. This measures the overlap between the bounding box generated by our model and the ground truth bounding box, represented as a percentage. This graph is measuring the percentage of correct bounding boxes and labels our model returned, with “correct” in this case referring to bounding boxes that had 50% or more overlap with their corresponding ground truth boxes. After training 17k steps, our model achieved 83% mean average precision.
source : https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193

Overfitting a lot to the train set ?
maybe yes, but inference will be done on very similar images (dashcam, etc...)

# Quantization
We need a scalable way to handle these inference requests with low latency. The output of a machine learning model is a binary file containing the trained weights of our model — these files are often quite large, but since we’ll be serving this model directly on a mobile device we’ll need to make it as small as possible.

This is where model quantization comes in. Quantization compresses the weights and activations in our model to an 8-bit fixed point representation. The following lines in our config file will generate a quantized model:
```
graph_rewriter {
  quantization {
    delay: 1800
    activation_bits: 8
    weight_bits: 8
  }
}
```
Typically with quantization, a model will train with full precision for a certain number of steps before switching to quantized training. The delay number above tells ML Engine to begin quantizing our weights and activations after 1800 training steps

source : https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193

also : https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/quantize#quantization-aware-training


# Running inference on embedded device with tensorflow Lite
We start by getting a TensorFlow frozen graph with compatible ops that we can use with TensorFlow Lite : run the export_tflite_ssd_graph.py script from the models/research directory with this command:
```
python object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=$CONFIG_FILE \
--trained_checkpoint_prefix=$CHECKPOINT_PATH \
--output_directory=$OUTPUT_DIR \
--add_postprocessing_op=true
```
source : https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193

for Google TPU Edge : https://coral.withgoogle.com/tutorials/edgetpu-retrain-detection/




If nothing printed on console add _tf.logging.set_verbosity(tf.logging.INFO)_ after imports in model_main.py


## ATTENTION :
anotations.csv lists my local path to images ...

Does not seem to impact tfrecords files, which have the following keys :
image/encoded
/filename
/format
/height
/key/sha256
/object/bbox/xmax
/object/bbox/xmin
/object/bbox/ymax
/object/bbox/ymin
/object/class/label ("1")
/object/class/text ("radar")
/object/difficult
/object/truncated
/object/view
/source_id
/width

also, th xml file of each image (in images folder) links to initial local folder
is it an issue ? apparently not,
see : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
A list of bounding boxes for the image. Each bounding box should contain:
"A bounding box coordinates (with origin in top left corner) defined by 4 floating point numbers [ymin, xmin, ymax, xmax]. *Note that we store the normalized coordinates (x / width, y / height) in the TFRecord dataset.*
The class of the object in the bounding box."


## TO DO :
1. OK tensorboard, or other way to capture loss / acc
2. OK error metric ?? : how to set it up ? : map & IOU in tensorboard
4. OK ## ATTENTION :
train.py is deprecated
should use model_main.py : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md#running-the-training-job
5. fine-tune a model w/ better accuracy
OR
5. train a better performing model for edge inference : https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html
6. OK ## ATTENTION ## :
anotations.csv lists my local path to images ...
also, th xml file of each image (in images folder) links to initial local folder
7. OK train with data augmentation
https://stackoverflow.com/questions/44906317/what-are-possible-values-for-data-augmentation-options-in-the-tensorflow-object
-> already some data augment specified in the config file :
data_augmentation_options:
random_horizontal_flip
ssd_random_crop
8. Run eveything on google cloud
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md
9. run everything on google colab, what is different from the followwing notebook ? (it's not an ssd)
https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/tf_TransferLearning.ipynb#scrollTo=uTR2xDZN9iwu
10. solutions if the model does not converge :
https://stackoverflow.com/questions/45633957/ssd-mobilenet-object-detection-algorithm-not-converging
change hyperparameters : batch_size, learning rate..
Adding more images will probably help
You want to see that the mAP has "lifted off" in the first few hours, and then you want to see when it converges. It's hard to tell without looking at these plots how many steps you need.
see here also : https://stackoverflow.com/questions/44973184/train-tensorflow-object-detection-on-own-dataset?noredirect=1&lq=1
