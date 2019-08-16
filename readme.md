## Code for ICCV 2019 Paper "[Mixed High-Order Attention Network for Person Re-Identification](http://bhchen.cn)"

This code is developed based on pytorch framework and the [baseline](https://github.com/layumi/Person_reID_baseline_pytorch) code.

* [Updates](#updates)
* [Files](#files)
* [Prerequisites](#prerequisites)
* [Train_Model](#train_model)
* [Evaluation](#evaluation)
* [Contact](#contact)
* [Citation](#citation)
* [LICENSE](#license)

### Updates
- Aug 16, 2019
  * The codes of training and testing for our [ICCV19 paper](http://bhchen.cn) are released.
  * We have cleared up and tested the codes on Market, Duke datasets, the expected retrieval performances are as follows:
  
  |Market | R@1 | R@5| R@10 | mAP | Reference |
  | -------- | ----- | ---- | ---- | ---- | ---- |
  | IDE+ERA | 89.9% | 96.4%| 97.6%| 75.6%|  `train_ide.py` |
  | IDE+MHN6 | 93.1%|97.7% |98.7%| 83.2% | `train_ide.py` |
  | PCB+ERA | 91.7%| 97.4%| 98.3%| 76.4% | `train_smallPCB` |
  | PCB+MHN4 | 94.3%| 98.0%| 98.8%| 83.9% | `train_smallPCB` |
  | PCB+MHN6 | 94.8%| 98.3%| 98.9%| 85.2% | `train_smallPCB_multiGPU.py` |
  
  |Duke | R@1 | R@5| R@10 | mAP | Reference |
  | -------- | ----- | ---- | ---- | ---- | ---- |
  | IDE+ERA | 82.7% |91.8%| 94.1%| 68.1%|  `train_ide.py` |
  | IDE+MHN6 | 87.8% |94.2%| 95.8%| 74.6% | `train_ide.py` |
  | PCB+ERA | 82.9%| 91.7%| 93.8%| 67.7% | `train_smallPCB` |
  | PCB+MHN4 | 88.5%| 94.5%| 96.1%| 76.9% | `train_smallPCB` |
  | PCB+MHN6 | 89.5%| 94.7%| 96.1%| 77.5% | `train_smallPCB_multiGPU.py` |

### Files
- train_ide.py test_ide.py

- train_smallPCB.py test_smallPCB.py

- train_smallPCB_multiGPU.py test_smallPCB.py

- auto_test.sh
### Prerequisites
* Pytorch(0.4.0+)
* python3.6
* 2GPUs, each > 11G
### Train_Model
1. The Installation is completely the same as [Caffe](http://caffe.berkeleyvision.org/). Please follow the [installation instructions](http://caffe.berkeleyvision.org/installation.html). Make sure you have correctly installed before using our code. 
2. Download the training images **CUB** {[google drive](https://drive.google.com/open?id=1V_5tS4YgyMRxUM7QHINn7aRYizwjxwmC), [baidu](https://pan.baidu.com/s/1X4W1xucDBxZafITvPF8SXQ)(psw:w3vh)} and move it to $(your_path). The images are preprossed the same as [Lifted Loss](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16/), i.e. with zero paddings.
3. Download the **training list**(400M) {[google drive](https://drive.google.com/open?id=1P2lUicV-nMchMU_aP6JbgzOibOG1o-F6), [baidu](https://pan.baidu.com/s/1-NH4rpkYwbLjR0tIkr30nA) (psw:xbct)}, and move it to folder "~/Hybrid-Attention-based-Decoupled-Metric-Learning-master/examples/CUB/". (Or you can create your own list by randomly selecting 65 classes with 2 samples each class.)
4. Download [googlenetV1](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel) (used for U512) and **3nets** {[google drive](https://drive.google.com/open?id=1boQISUyXaV77qCS0u5Nmlv6dckuN25mM), [baidu](https://pan.baidu.com/s/10Q1wPeHMYtXEJ5cqY9GgPg)(psw:dmev)} (used for DeML3-3_512) to folder "~/Hybrid-Attention-based-Decoupled-Metric-Learning-master/examples/CUB/pre-trained-model/". (each stream of 3nets model is intialized by the same googlenetV1 model)
5. Modify the images path by changing "root_folder" into $(your_path) in all *.prototxt .
6. Then you can train our baseline method **U512** and the proposed **DeML(I=3,J=3)** by running
```
        cd ~/Hybrid-Attention-based-Decoupled-Metric-Learning-master/examples/CUB/U512
        ./finetune_U512.sh
```     
and
```
        cd ~/Hybrid-Attention-based-Decoupled-Metric-Learning-master/examples/CUB/DeML3-3_512
        ./finetune_DeML3-3_512.sh
```     
the trained models are stored in folder "run_U512/" and "run_DeML3-3_512/" respectively.
### Extract_DeepFeature
1. run the following code for **U512** and **DeML(I=3,J=3)** respectively.
```
        cd ~/Hybrid-Attention-based-Decoupled-Metric-Learning-master/examples/CUB/U512
        ./extractfeatures.sh
```

```
        cd ~/Hybrid-Attention-based-Decoupled-Metric-Learning-master/examples/CUB/DeML3-3_512
        ./extractfeatures.sh
```
the feature files are stored at folder "features/"
### Evaluation
1. Run code in folder "~/Hybrid-Attention-based-Decoupled-Metric-Learning-master/evaluation/"
### Contact 
- [Binghui Chen](http://bhchen.cn)

### Citation
You are encouraged to cite the following papers if this work helps your research. 

    @inproceedings{chen2019hybrid,
      title={Hybrid-Attention based Decoupled metric Learning for Zero-Shot Image Retrieval},
      author={Chen, Binghui and Deng, Weihong},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2019},
    }
    @InProceedings{chen2019energy,
    author = {Chen, Binghui and Deng, Weihong},
    title = {Energy Confused Adversarial Metric Learning for Zero-Shot Image Retrieval and Clustering},
    booktitle = {AAAI Conference on Artificial Intelligence},
    year = {2019}
    }
    @inproceedings{songCVPR16,
    Author = {Hyun Oh Song and Yu Xiang and Stefanie Jegelka and Silvio Savarese},
    Title = {Deep Metric Learning via Lifted Structured Feature Embedding},
    Booktitle = {Computer Vision and Pattern Recognition (CVPR)},
    Year = {2016}
    }
### License
Copyright (c) Binghui Chen

All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

***

#Market
IDE+ERA           89.9 96.4 97.6 mAP75.6
IDE_MHN6 alpha1.4 93.1 97.7 98.7 mAP83.2

PCB+ERA           91.7 97.4 98.3 mAP76.4
PCB_MHN4 alpha2   94.3 98.0 98.8 mAP83.9
PCB_MHN6 alpha2   94.8 98.3 98.9 mAP85.2

#Duke
IDE+ERA           82.7 91.8 94.1 mAP68.1
IDE_MHN6 alpha1.4 87.8 94.2 95.8 mAP74.6

PCB+ERA           82.9 91.7 93.8 mAP67.7
PCB_MHN4 alpha2   88.5 94.5 96.1 mAP76.9
PCB_MHN6 alpha2   89.5 94.7 96.1 mAP77.5
