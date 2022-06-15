# DM-HW1
The homework aims to identify the smoking status of the patient by using machine learning approaches.

## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04 LTS
- Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz
- NVIDIA RTX 2070

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended. {envs_name} is the new environment name which you should assign.
```bash
conda create -n {envs_name} python=3.6
source activate {envs_name}
pip install -r requirements.txt
```

## Training
```bash
$ python3 train.py
```

## Details
The code, train.py, using 2-layers bidirectional LSTM model to train and predict the smoking status of patients.

The code will output three files
1. loss_graph.png: shows the loss during training
2. acc_graph.png: shows the accuracy during training and validation
3. case1_5.txt: records prediction of test data
