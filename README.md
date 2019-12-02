# Comp_ReID-master
Code for 2019 NAIC Person ReID Stage.1

# Installation
* python==3.6.9
* torch==0.4.1
* torchvision==0.2.2
* numpy==1.17.3
* Pillow==6.2.1

# Preparation
1. Run `git clone https://github.com/SWRDFK/Comp_ReID-master.git`
2. Prepare dataset: 
    download the competition datasets and make sure the directory as following:    
    &emsp;|—— dataset/  
    &emsp;|&emsp;&emsp;&ensp;|—— gallery_b/  
    &emsp;|&emsp;&emsp;&ensp;|—— query_b/  
    &emsp;|&emsp;&emsp;&ensp;|—— train_set/  
    &emsp;|&emsp;&emsp;&ensp;|—— train_list.txt  
3. Download ResNet101_ibn_a pretrained models from the following url and put it under the folder  
`$Comp_ReID-master/core/pretrained`  
   链接: https://pan.baidu.com/s/1935MdSvnS1t6qo9TH-nXcQ 提取码: jk3d

# Train
You can train the following models respectively.

## Train model A (resnet101a_SA)
Run `python main.py --mode train --model_name resnet101a_SA`

## Train model B (resnet101a_RLL)
Run `python main.py --mode train --model_name resnet101a_RLL`

## Train model C (densenet161_CBL)
Run `python main.py --mode train --model_name densenet161_CBL`

# Test
You can download dists, jsons and models from the following url and put it under the folder `$Comp_ReID-master/output`  
链接: https://pan.baidu.com/s/1sNZf2WD895KsFkh6HrhkSA 提取码: x5k6

After training, you can test with your trainde models or directly use our models.  

After testing, it will generate two files for each model:  
&emsp;1. the distance matrix between query and gallery named `"model_name".npy`, saved in `$Comp_ReID-master/output/dists`.  
&emsp;2. the uploading json file for evaluation named `"model_file".json`, saved in `$Comp_ReID-master/output/jsons`.  

## Test model A (densenet161_CBL)  
Run `python main.py --mode test --model_name densenet161_CBL`

## Test model B (resnet101a_RLL)  
Run `python main.py --mode test --model_name resnet101a_RLL`

## Test model C (resnet101a_SA)  
Run `python main.py --mode test --model_name resnet101a_SA`

# Ensemble
You can test the ensemble model by using three distance matrices and get the generated `ensemble.json` which saved in `$Comp_ReID-master/output/jsons`.  
Run `python main.py --mode ensemble`
