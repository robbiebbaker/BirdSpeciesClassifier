# Image Classification of birds

**A whistle stop tour of how to use the latest image classification techniques to build and understand a deep neural network bird classifier**

This is an investigation using PyTorch CNNs of deep image classificaton to solve a bird species classification problem with the [Caltech UCSD Birds dataset (CUB-200-2011)](http://www.vision.caltech.edu/visipedia/CUB-200.html).

## Problem Statement
Building an optimal Bird Species Classifier.

## Dataset details

### Dataset
Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version of the CUB-200 dataset, with roughly double the number of images per class and new part location annotations. For detailed information about the dataset, please see the technical report linked below.

#### Number of categories
* 200 catergories
#### To name a few
* Black_footed_Albatross
* Laysan_Albatross
* Sooty_Albatross
* Parakeet_Auklet
* Rhinoceros_Auklet
* Brewer_Blackbird

#### Number of images
* 11,788

#### List of attributes
Total 312 binary attributes
To name a few
* has_bill_shape::curved_(up_or_down)
* 10 has_wing_color::blue
* has_upperparts_color::blue
* has_breast_pattern::spotted
* has_back_color::brown
* has_tail_shape::forked_tail


### Support Vector Machines

#### Description  
A support vector machine builds a hyperplane or set of hyperplanes in a high- or infinite dimensional space, used for classification. Good separation is achieved by the hyper plane that has the largest distance to the nearest training data point of any class (functional margin), generally larger the margin lower the generalization error of the classifier.

#### Characteristics 
SVM uses Nonparametric with binary classifier approach and can handle more input data very efficiently. Performance and accuracy depends upon the hyperplane selection and kernel parameter.


### Convolutional Neural Network

#### Description 
* CNN compare images piece by piece. By finding approximate matching features in same positions in the two images, it’s performs better those schemes that simply matching the two images as a whole. 
* Filtering is used to match features with the image patch. The convolutional aspect involves repetitive matching in every possible way. Each image becomes a stack of filtered images. ReLU is a used to normalise the values. 
* Pooling involves shrinking the stack size by taking strides of fixed length and picking the highest values in each window. The above process is repeated many times to form deep stacking of layers. 
* Backpropagation could be used to improve accuracy over iterations, and for this we need a collection of images for which we know the answer.

#### Characteristics 
The hyper parameters that the user has to feed to the network are
##### Convolution 
* Number of features
* Size of features
##### Pooling
* Window size
* Window stride
##### Fully Connected
* Number of neurons.

An inception model is a CNN which uses convolution kernels of multiple sizes as well as pooling within one layer. As it turns out this yields very good results because hierarchical information with varying sizes can be identified at the same layer. Plus the fact that you don't have to worry about setting kernel sizes so rigidly because many different size are used. It is always good to have a general rule like this instead of a hyper parameter.

### Design of model
* Transfer Learning
It is a technique in which a model trained on one task is purposed on another task. Transfer learning is an optimization that allows rapid progress or improved performance when modeling the second task.

#### How to Use Transfer Learning?
You can use transfer learning on your own predictive modeling problems. Two common approaches are as follows:
* Develop Model Approach
* Pre-trained Model Approach

#### Pre-trained Model Approach
* Select Source Model. 
A pre-trained source model is chosen from available models. Many research institutions release models on large and challenging datasets that may be included in the pool of candidate models from which to choose from.
* Reuse Model. 
The model pre-trained model can then be used as the starting point for a model on the second task of interest. This may involve using all or parts of the model, depending on the modeling technique used. 
* Tune Model. 
Optionally, the model may need to be adapted or refined on the input-output pair data available for the task of interest.
One of the techniques in our project involved Transfer Learning with Image Data. So for we used pre-trained ImageNet model, for running *Inception Classification technique 


# Change Log

### Installing a python training environment

An environment file for installing the runtime environment has also been provided.

**This should be run on a machine with the relevant version of the CUDA drivers installed, at current time of writing 11.1. To change the CUDA version, ensure that the cudatoolkit version is correct for the version you have installed, and also check the PyTorch version**. 

It is currently recommended that PyTorch be installed through Conda, as the PYPI versions were not working correctly (_at the time of writing, v1.8.1 when installed through PYPI was causing some computation errors due to linking errors with backend libraries. This was not happening when using Conda installed depdendencies_).

To install an environment using the conda dependencies file, run:

```shell
conda env create -f conda_depdencies.yml
```

## V2.0 Refactoring of training scripts to use YACS YAML configuration files

Training scripts have been completely refactored to use a new Trainer class implementation.

### YACS YAML config files

All configuration and parameters for the model training is now done through YAML configuration files to specify the model configuration.

Examples can be found _scripts/configs_ for the 3 model libraries that are supported by the _cub_tools_ package.

For a complete list of configuration fields, see _pkg/cub_tools/config.py_.

```yaml
# my_project/config.py

MODEL:
    MODEL_LIBRARY: 'timm'
    MODEL_NAME: 'resnext101_32x8d'
DIRS:
    ROOT_DIR: '/home/edmorris/projects/image_classification/caltech_birds'
TRAIN:
    NUM_EPOCHS: 40
```

### New training scripts

Models are trainined by calling the training script and specifying the configuration script.

To train a ResNeXt101_32x8d from the TIMM library, the command run from the scripts directory would be as follows:

```shell
python train_pytorch_caltech_birds.py --config configs/timm/resnext101_32x8d_config.yaml
```

An addition to this release is the integration of PyTorch Ignite training framework to give additional training information. The following features have been implemented.
   1. **Tensorboard logging**. To monitor, simply start Tensorboard with the logging directory watching the models/ directory.
   2. **Model checkpointing**. Saves out the best model after each iteration.
   3. **Early Stopping**. Terminates training if the loss does not improve after a specified number of iterations.

To use these extra features enabled by the PyTorch Ignite framework, run the following:

```shell
python train_pytorch_ignite_caltech_birds.py --config configs/timm/resnext101_32x8d_config.yaml
```

The open-source MLOps system ClearML enables the logging and remote execution of machine learning experiments. It logs everything about a model training, including code status (from the repository), environment dependencies, data and model versions and also training metrics and outputs in an easy to use web based GUI. A version of the training script has been developed to be able to deploy model training via the ClearML system. Note, this version uses Ignite as the framework integrates with ClearML to log training metrics out of the box.

```shell
python train_clearml_pytorch_ignite_caltech_birds.py --config configs/timm/resnext101_32x8d_config.yaml
```

## V1.0 Initial Release

Release of the workshop material with per model training scripts.


# Dataset details and data

## Dataset

Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version of the CUB-200 dataset, with roughly double the number of images per class and new part location annotations. For detailed information about the dataset, please see the technical report linked below.

Number of categories: 200

Number of images: 11,788

Annotations per image: 15 Part Locations, 312 Binary Attributes, 1 Bounding Box

Some related datasets are Caltech-256, the Oxford Flower Dataset, and Animals with Attributes. More datasets are available at the Caltech Vision Dataset Archive.

## Data files

To download the data click on the following links:

   1. Images and annotations [CalTech Visipedia Website - CUB_200_2011.tgz](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   
   **The image tarball should be downloaded to repository root, and extracted into the data/images_orig sub-directory, as shown below.***
               
   2. Segmentations (optional, not need for this work) [CalTech Visipedia Website - segmentations.tgz](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz)
    
Place the files into the root of the cloned caltech_birds repo file structure.

Unzip the dowloaded zip files into cloned repository structure that replicates the project structure:

    caltech_birds-|
        data-|   ** SEPARATE DOWNLOAD FROM CALTECH VISIPEDIA WEBSITE **
            attributes-|
            images_orig-|   ** RENAME IMAGES FOLDER TO IMAGE_ORIG **
               ##class_name##-|
                  class001-image001.jpg
                  class001-image002.jpg
            images-|   ** NEW FOLDER CREATED BY FIRST WORKFLOW PROCESS **
               train-|
                  ##class_name##-|
                     class###-image001.jpg
               test-|
                  ##class_name##-|
                     class###-image035.jpg
            parts-|
            attributes.txt
            bounding_boxes.txt
            classes.txt
            image_class_labels.txt
            images.txt
            README
            train_test_split.txt
        example-notebooks-|   ** WORKSHOP MATERIAL IS IN HERE **
        models-|   ** SEPARATE DOWNLOAD FROM RELEASES **
            classification-|
                #modelname1#-|
                #modelname2#-|
        notebooks-|      ** INVESTIGATIONS OF ALL TYPES OF CNN ARCHITECTURES ON THE BIRD CLASSIFICATION PROBLEM **
        pkg-|            ** CUB_TOOLS PYTHON PACKAGE. INSTALL FROM IN HERE.
        scripts-|        ** TRAINING SCRIPTS FOR PYTORCH CNN MODELS **

# Installation

# Additional files
        
**notebooks** directory contain the Jupyter notebooks where the majority of the visualisation and high level code will be maintained.

**scripts** directory contains the computationally intensive training scripts for various deep CNN architectures, which take longer to run (even on GPU) and are better executed in a python script. I suggest this be done using some form of terminal persistance method (if running in the cloud) to keep the terminal session open whilst the model is training, allowing you to log off the remote host without killing the process. These can typically take a few hours, to a day or so to complete the prescribed epochs. I prefer to use [TMUX](https://github.com/tmux/tmux/wiki/Getting-Started), which is a Linux utility that maintains separate terminal sessions that can be attached and detached to any linux terminal session once opened. This way, you can execute long running python training scripts inside a TMUX session, detach it, close down the terminal session and let the process run. You can then start a new terminal session, and then attach the running TMUX session to view the progress of the executed script, including all its output to terminal history.

**pkg** directory contains all the utility functions that have been developed to process, visualise, train and evaluate CNN models, as well results post processing have been contained. It has been converted into a python package that can be installed in the local environment by running in the **pkg** directory, *pip install -e .*. The functions can then be accessed using the cub_tools module import as ***import cub_tools***.

**models** directory contains the results from the model training processes, and also any other outputs from the evaluation processes including model predictions, network feature maps etc. **All model outputs used by the example notebooks can be downloaded from the release folder of the Github repo.** The models zip should be placed in the root of the repo directory structure and unziped to create a models directory with the ResNet152 results contained within. Other models are also available including PNASNET, Inception V3 and V4, GoogLenet and ResNeXt variants.
