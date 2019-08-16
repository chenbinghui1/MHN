# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from model import PCB, MHN_smallPCB
from PIL import Image

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/home/zzd/Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--parts', default=4, type=int, help='Order, the max value is 4, for higher order, please use train_mhn_smallPCB_multi.py')
parser.add_argument('--mhn', action='store_true', help='Mixed High Order' )

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#

data_transforms = transforms.Compose([
        transforms.Resize((336,168), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])


data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        #print(count)
        if not opt.mhn:
            ff = torch.FloatTensor(n,256,part).zero_()
        else:
            ff = torch.FloatTensor(n,256,part*parts).zero_()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            if opt.mhn:
                _, fea = model(input_img)
                for j in range(part):
                    for k in range(parts):
                        f = fea[j][k*n:(k+1)*n,:].data.cpu()
                        ff[:,:,j*parts+k] = ff[:,:,j*parts+k] + f
            else:
                outputs = model(input_img) 
                for j in range(part):
                    f = outputs[j].data.cpu()
                    ff[:,:,j] = ff[:,:,j] + f

        # norm feature
        
        # feature size (n,2048,6)
        # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        if not opt.mhn:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(part) 
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(part*parts) 
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
        

        features = torch.cat((features,ff), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

if opt.multi:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_cam,mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
class_number={'datasets/Market/datasets/pytorch/':751,'datasets/Duke/datasets/pytorch/':702,'datasets/CUHK03_detected/datasets/pytorch/':767,'datasets/CUHK03_labled/datasets/pytorch/':767}
class_num = class_number[data_dir]
part = 3
if not opt.mhn:
    model_structure = PCB(class_num,part)
else:
    parts = opt.parts
    model_structure = MHN_smallPCB(class_num, parts, part)
    model_structure.model.fc = nn.Sequential()
    model_structure.model.avgpool = nn.Sequential()
#print(model_structure)


model = load_network(model_structure)## load the learned params
if not opt.mhn:
    model.model.avgpool = nn.Sequential()
    model.model.fc = nn.Sequential()
    # Remove the final fc layer and classifier layer
    model.classifier0.classifier = nn.Sequential()
    model.classifier1.classifier = nn.Sequential()
    model.classifier2.classifier = nn.Sequential()
    #model.classifier3.classifier = nn.Sequential()
    #model.classifier4.classifier = nn.Sequential()
    #model.classifier5.classifier = nn.Sequential()

# Change to test mode
model = model.eval()

if use_gpu:
    model = model.cuda()


##### Extract feature
gallery_feature = extract_feature(model,dataloaders['gallery'])
query_feature = extract_feature(model,dataloaders['query'])
if opt.multi:
    mquery_feature = extract_feature(model,dataloaders['multi-query']) 
##### Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result{}.mat'.format(gpu_ids[0]),result)
if opt.multi:
    result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
    scipy.io.savemat('multi_query.mat',result)
