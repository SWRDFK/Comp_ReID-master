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
    1. download the competition datasets into Comp_ReID-master/dataset, which contains 4 subfiles:
    
        -- Comp_ReID-master/dataset/train_set
        
        -- Comp_ReID-master/dataset/train_list.txt
        
        -- Comp_ReID-master/dataset/query_b
        
        -- Comp_ReID-master/dataset/gallary_b
    2. make sure the data folder like the following structure:  
    `Comp_ReID-master/dataset`
    3. ResNet101_ibn_a pretrained(链接: https://pan.baidu.com/s/1PpcKc2Ji5joZPxG8TE2DgQ 提取码: 7e69) model.
# Train
We train the following models respectively.

## Train model A (resnet101a_SA)

Uncomment line 58 in core/base.py and choose the resnet101a_SA model. Then run

`python main.py --mode train --model_name resnet101a_SA`

## Train model B (resnet101a_RLL)

Uncomment line 55 in core/base.py and choose the resnet101a_RLL model. Then run

`python main.py --mode train --model_name resnet101a_RLL`

## Train model C (densenet161_CBL)

Uncomment line 61 in core/base.py and choose the resnet101a_RLL model. Then run

`python main.py --mode train --model_name densenet161_CBL`

# Test

The experiment results are generated at dists/ file. And it will generate two files. One is the distance matrix "model_name".npy and the other is the upload file for this competition "model_file".json.

## Test model A (resnet101a_SA)
`python main.py --mode test --model_name resnet101a_SA`
## Test model B (resnet101a_RLL)
`python main.py --mode test --model_name resnet101a_RLL`
## Test model C (densenet161_CBL)
`python main.py --mode test --model_name densenet161_CBL`

## Test ensemble model
`python main.py --mode ensemble`
You can also directly download the trained models from (链接: https://pan.baidu.com/s/1eCGHenIsi_eh1rxfbz0-yw 提取码: fzru).

If you want to conduct ensemble experiment, make sure your outputfile has aboving three pretrained models. And the file structure is as:

./output

    --densenet161_CBL
  
        --logs
    
        --models
    
    --resnet101a_RLL
  
        --logs
    
        --models
    
    -- resnet101a_SA
  
        --logs
    
        --models
        
We also provide our ensemble results as:

dists(链接: https://pan.baidu.com/s/1TukolUhlI8R8VhNxvlsB8g 提取码: bxi4)

jsons(链接: https://pan.baidu.com/s/103ct1HvzQk8k3v2h8XNwFA 提取码: 9xrd)
