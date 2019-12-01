# Comp_ReID-master
Code for 2019 NAIC Person ReID
# Installation
* python==3.6.9
* torch==0.4.1
* torchvision==0.2.2
* numpy==1.17.3
* Pillow==6.2.1
# Preparation
1. Run `git clone https://github.com/SWRDFK/Comp_ReID-master.git`
2. Prepare dataset: 
    1. download the competition datasets
    2. make sure the data folder like the following structure:  
    `Comp_ReID-master/dataset`
    3. ResNet101_ibn_a pretrained(链接: https://pan.baidu.com/s/1PpcKc2Ji5joZPxG8TE2DgQ 提取码: 7e69) model.
# Train
We train the following models respectively.

## Train model A (resnet101a_SA)
`python main.py --mode train --model_name resnet101a_SA`

## Train model B (resnet101a_RLL)
`python main.py --mode train --model_name resnet101a_RLL`

## Train model C (densenet161_CBL)
`python main.py --mode train --model_name densenet161_CBL`

# Test
You can directly download the trained models from (链接: https://pan.baidu.com/s/153bRn2n1z9nHVEdE3kDgOg 提取码: grwi).

## Test model A (resnet101a_SA)
`python main.py --mode test --model_name resnet101a_SA`
## Test model B (resnet101a_RLL)
`python main.py --mode test --model_name resnet101a_RLL`
## Test model C (densenet161_CBL)
`python main.py --mode test --model_name densenet161_CBL`

## Test ensemble model
`python main.py --mode ensemble`

dists(链接: https://pan.baidu.com/s/1TukolUhlI8R8VhNxvlsB8g 提取码: bxi4)

jsons(链接: https://pan.baidu.com/s/103ct1HvzQk8k3v2h8XNwFA 提取码: 9xrd)
