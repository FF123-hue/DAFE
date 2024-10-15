# DAFE
Code of paper "A Dual-Correlation Feature Enhancement Network Model Based on Transformer for Occluded Pedestrian Re-identification" 

A simple but effective method for Occluded Person Re-identification

# Requirements
# Installation
pip install -r requirements.txt

(we use /torch 1.6.0 /torchvision 0.7.0 /timm 0.3.2 /cuda 10.1 / 16G V100 for training and evaluation. Note that we use torch.cuda.amp to accelerate speed of training which requires pytorch >=1.6)

# Prepare Datasets
mkdir data

#Occluded-Duke dataset 

https://github.com/lightas/Occluded-DukeMTMC-Dataset

#Market-1501 dataset 

https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ

Then unzip them and rename them under the directory like

# Training on Occluded-Duke
python train.py --config_file configs/OCC_Duke/vit_transreid_stride.yml 

# Training on Market-1501
python train.py --config_file configs/Market/vit_transreid_stride.yml 

# Testing on Occluded-Duke
python test.py --config_file configs/OCC_Duke/vit_transreid_stride.yml  TEST.WEIGHT 'log/lr0008_b32/transformer_best.pth'
