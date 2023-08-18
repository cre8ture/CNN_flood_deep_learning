# CS7643_Group_Project

Author: Chanyu Yang (cyang490@gatech.edu), Kai Kleinbard (kkleinbard3@gatech.edu)

## Introduction
This folder includes data and files needed to conduct data preprocessing, model training, validation and testing, as well as visualisation. 

The link of this folder is: https://drive.google.com/drive/folders/1AmVJ9qA7q3KNZvEGW9KVfCKWzXDmnBO6?usp=sharing


## environment
Please activate the environment before running the commands in the following sections. To create the environment using .yml file, type the following command:

```
~/CS7643_Group_Project$ conda env create -f environment.yml
```

Please refer to [conda website](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more details of environment management.

## data
This subfolder includes the flow records of the two upstream gauges (see ```*_15min.csv```), files needed to run LISFLOOD-FP (see ```./data/lisflood_resuls/```), and preprocessed data that are ready for running the deep learning model (```*matrix.csv```).

### how to run lisflood-fp
To generate flood water depths, the hydrodynamic model LISFLOOD-FP is available upon request. Please check the [website](http://www.bristol.ac.uk/geography/research/hydrology/models/lisflood/). Once the LISFLOOD-FP is ready, put the ```lisflood```executable file into the ```./data/lisfood_results/```subfolder. Within this subfolder, run the following command:

```
~/CS7643_Group_Project$ ./data/lisflood_results/lisflood -v <name_of_your_file>.par
``` 

For more information, please check the attached user manual when requesting LISFLOOD-FP. 


## flow
This subfolder includes the programs that can reproduce graphs, preprocessed data that can be used directly by the LISFLOOD-FP and the deep learning model.

### how to use
```
~/CS7643_Group_Project$ python ./flow/test_hydro_gauge.py
```
```
~/CS7643_Group_Project$ python ./flow/test_preprocess4dl.py
```

## lisflood
This subfolder includes the programs that can reproduce graphs, preprocessed data that can be used directly by the deep learning model.

### how to use
```
~/CS7643_Group_Project$ python ./lisflood/test_lisflood_analysis.py
```

## Deep learning model
The ```model.py``` contains the model architecture. ```get_data.py``` is the module that get the input and target data and convert them from csv to tensor format that can be run using pytorch. ```train_model.py``` is the module that is used for training and generate relevant graphs. 

### how to use

```
~/CS7643_Group_Project$ python ./train_model.py
```