
#import 
import os
import argparse
import json
import shutil
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from dataset import Talk2Car
from utils.collate import custom_collate
from utils.util import AverageMeter, ProgressMeter, save_checkpoint

import models.resnet as resnet
import models.nlp_models as nlp_models


import numpy as np
from PIL import Image
from utils.vocabulary import Vocabulary
from utils.math import jaccard
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
parser = argparse.ArgumentParser(description='Talk2Car object referral')
parser.add_argument('--root', metavar='DIR',
                    help='path to dataset',default='./data')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=18, type=int,
                    metavar='N',
                    help='mini-batch size (default: 18)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--milestones', default=[4, 8], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--nesterov', action='store_true',
                    help='nesterov')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')



args = parser.parse_args()

# Create dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
train_dataset = Talk2Car(root=args.root, split='train',
                            transform=transforms.Compose([transforms.ToTensor(), normalize]))
val_dataset = Talk2Car(root=args.root, split='val',
                    transform=transforms.Compose([transforms.ToTensor(), normalize]))

train_dataloader = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True,
                        num_workers=args.workers, collate_fn=custom_collate, pin_memory=True,                            drop_last=True)
val_dataloader = data.DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=custom_collate, pin_memory=True,                            drop_last=False)


# Create model
img_encoder = resnet.__dict__['resnet18'](pretrained=True) 
text_encoder = nlp_models.TextEncoder(input_dim=train_dataset.number_of_words(),
                                                hidden_size=512, dropout=0.1)


criterion = nn.CrossEntropyLoss(ignore_index = train_dataset.ignore_index, 
                                reduction = 'mean')   

cudnn.benchmark = True

# Optimizer and scheduler
params = list(text_encoder.parameters()) + list(img_encoder.parameters())
optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, 
                        weight_decay=args.weight_decay, nesterov=args.nesterov)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                        gamma=0.1)

# Checkpoint: load the model trained before:
checkpoint = 'checkpoint.pth.tar'
if os.path.exists(checkpoint):
    checkpoint = torch.load(checkpoint, map_location='cpu')
    img_encoder.load_state_dict(checkpoint['img_encoder'])
    text_encoder.load_state_dict(checkpoint['text_encoder'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch'] 
    best_ap50 = checkpoint['best_ap50']
else:
    print("=> no checkpoint at %s" %(checkpoint))
    best_ap50 = 0
    start_epoch = 0
    

def image(num):
    # start :    
    # all the image is after clipping with mean and sdv need to recover for better visulization.
    num=num
    split = 'train'
    dataset = Talk2Car(args.root, split, './utils/vocabulary.txt', transforms.ToTensor())
    sampleA = dataset.__getitem__(num)
    imgA = np.transpose(sampleA['image'].numpy(), (1,2,0))
    #print(img)
    #print('=> Plot image ')
    fig, ax = plt.subplots(1)
    ax.imshow(imgA)
    plt.axis('off')
    plt.savefig('./static/images/demo.png')


def predict(command,num):
    split = 'train'
    dataset = Talk2Car(args.root, split, './utils/vocabulary.txt', transforms.ToTensor())
    sampleA = dataset.__getitem__(num)
    imgA = np.transpose(sampleA['image'].numpy(), (1,2,0))

    dataset = train_dataset
    # Choose image :
    sample = dataset.__getitem__(num)
    
    img_encoder.eval()
    text_encoder.eval()
    region_proposals = sample['rpn_image']
    # Command input :
    Acommand = command
    
    # Figure vocabulary :
    vocabulary='./utils/vocabulary.txt'
    vocabulary = Vocabulary(vocabulary)
    
    # Coommand process:
    command = vocabulary.sent2ix_andpad(Acommand, add_eos_token=True)
    command = torch.LongTensor(command)
    command_length=torch.LongTensor([len(vocabulary.sent2ix(Acommand)) + 1])
    command_length = command_length
    
    # Region of proposals:
    r, c, h, w = region_proposals.size()
    b=1
    
    # Image features
    img_features = img_encoder(region_proposals.view(b*r, c, h, w))
    norm = img_features.norm(p=2, dim=1, keepdim=True)
    img_features = img_features.div(norm)
   
    # Sentence features
    #print(command.size())
    new= torch.reshape(command,(58,1))
    _, sentence_features = text_encoder(new, command_length)
    norm = sentence_features.norm(p=2, dim=1, keepdim=True)
    sentence_features = sentence_features.div(norm)
 
    # Product in latent space
    scores = torch.bmm(img_features.view(b, r, -1), sentence_features.unsqueeze(2)).squeeze()
    
    #Top k choice coz model 
    #preds=torch.topk(scores,3)[1]
    
    # best choice :
    
    pred = torch.argmax(scores)
    output=region_proposals[pred]
    
    
    # draw a box in the imagine :
    fig, ax = plt.subplots(1)
    ax.imshow(imgA)
    bbox=sample['rpn_bbox_lbrt'][pred].tolist()
    xl, yb, xr, yt = bbox 
    #print(bbox)
    w, h = xr - xl, yt - yb
    rect = patches.Rectangle((xl, yb), w, h, fill = False, edgecolor = 'r')
    ax.add_patch(rect)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./static/images/demo.png')
 

