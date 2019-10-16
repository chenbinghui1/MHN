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
from PIL import Image
import time
import os
from model import PCB, MHN_smallPCB
from random_erasing import RandomErasing
import json
from shutil import copyfile
from torch.nn import init
import losses
from own_sampler import BalancedBatchSampler
version =  torch.__version__
############  fix the seed ##############
np.random.seed(2019)
torch.manual_seed(2018)
torch.cuda.manual_seed(2017)
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir',default='/home/zzd/Market/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--alpha', default=1.0, type=float, help='div loss weights')
parser.add_argument('--parts', default=4, type=int, help='Order, the max value is 4, for higher order, please use train_mhn_smallPCB_multi.py')
parser.add_argument('--balance_sampler', action='store_true', help='use balance_sampler' )
parser.add_argument('--mhn', action='store_true', help='Mixed High Order' )
opt = parser.parse_args()

data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
#print(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
balance_sampler_ = False
transform_train_list = [
        transforms.Resize((336,168), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
transform_val_list = [
        transforms.Resize(size=(336,168),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}


train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

if not opt.balance_sampler:
    balance_sampler_ = False
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8) # 8 workers may work faster
                  for x in ['train', 'val']}
else :     
    balance_sampler_ = True                                   
    batch_sampler = {}
    batch_sampler['train'] = BalancedBatchSampler(image_datasets['train'], n_classes = opt.batchsize/2, n_samples = 2)
    batch_sampler['val'] = BalancedBatchSampler(image_datasets['val'], n_classes = opt.batchsize/2, n_samples = 2)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_sampler = batch_sampler[x],
                                             num_workers=8)# 8 workers may work faster
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time()-since)
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

#### for cosmargin loss #####
#def weights_init_linear(m):
#    init.normal_(m.weight.data, std=0.001)
#    init.constant_(m.bias.data, 0.0)
class_number={'datasets/Market/datasets/pytorch/':751,'datasets/Duke/datasets/pytorch/':702,'datasets/CUHK03_detected/datasets/pytorch/':767,'datasets/CUHK03_labled/datasets/pytorch/':767}
feature_dim = 512
class_num = class_number[data_dir]

#############################
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step() 
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            div_loss = 0.0
            cnt = 0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
                cnt += 1
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                
                #print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if not opt.mhn:
                    outputs = model(inputs)
                    ##############################
                    sm = nn.Softmax(dim=1)
                    num_part = part
                    score = sm(outputs[0])
                    for i in range(num_part-1):
                        score += sm(outputs[i+1])
                    _, preds = torch.max(score.data, 1)
                    loss = criterion(outputs[0], labels)
                    for i in range(num_part-1):
                        loss += criterion(outputs[i+1], labels)
                else:
                    y, fea = model(inputs)
                    ###############################
                    sm = nn.Softmax(dim=1)

                    score = sm(y[0])
                    num_part = part # 6
                    for i in range(num_part):
                        for j in range(parts):
                            if i+j == 0:
                                continue
                            score += sm(y[i*parts+j])

                    _, preds = torch.max(score.data, 1)

                    loss_sm = criterion(y[0], labels)
                    for i in range(num_part):
                        for j in range(parts):
                            if i+j == 0:
                                continue
                            loss_sm += criterion(y[i*parts+j], labels)
                    loss_sm = loss_sm / (parts)
                    loss_div = criterion_div(fea[0])
                    for i in range(num_part-1):
                        loss_div += criterion_div(fea[i+1])
                    loss_div /= num_part
                    loss = loss_sm + loss_div * opt.alpha
                    div_loss += loss_div.item()
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(model.parameters(),20)
                    optimizer.step()

                # statistics
                if int(version[2]) > 3 or int(version[0]) == 1: # for the new version like 0.4.0 and 0.5.0
                    running_loss += loss.item() * now_batch_size
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            if phase == 'train' and opt.mhn:
                print("diverse loss: ",div_loss / cnt)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)            
            # deep copy the model
            last_model_wts = model.state_dict()
            if epoch%10 == 9:
                save_network(model, epoch)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model



######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.module.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

part = 3 # smallPCB
if not opt.mhn:
    model = PCB(len(class_names),part)
else:
    parts = opt.parts
    model = MHN_smallPCB(len(class_names), opt.parts, part)# For multi-gpu, order 6 can be implemented
    criterion_div = losses.AdvDivLoss(parts=parts)
    model.model.avgpool = nn.Sequential()
    model.model.fc = nn.Sequential()
    
print(model)
model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()

if use_gpu:
    model = model.cuda()
    if opt.mhn:
        criterion_div.cuda()

criterion = nn.CrossEntropyLoss()
print("balance_sampler_ : ",balance_sampler_)
ignored_params = list(map(id, model.module.model.parameters() ))       
base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
if not opt.mhn:
    optimizer_ft = optim.SGD([
             {'params': model.module.model.parameters(), 'lr': 0.01},
             {'params': base_params, 'lr': 0.1}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
    optimizer_ft = optim.SGD([
             {'params': model.module.model.parameters(), 'lr': 0.01},
             {'params': base_params, 'lr': 0.1},
             {'params': criterion_div.parameters(), 'lr': 0.01}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 6 hours on GPU. 
#
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
#record every run
copyfile('./train_smallPCB_multiGPU.py', dir_name+'/train_smallPCB_multiGPU.py')
copyfile('./model.py', dir_name+'/model.py')

# save opts
with open('%s/opts.json'%dir_name,'w') as fp:
    json.dump(vars(opt), fp, indent=1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=70)

