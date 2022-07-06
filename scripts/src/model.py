import torch
import torch.nn as nn
import torchvision.models as models
from conf import *

def build_model():
    if args.model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

    elif args.model_name == 'resnet18':
        model = models.resnet18(pretrained=True)

    #Modify the classifier for agriculture data
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs,512),
                            nn.ReLU(),
                            nn.Dropout(p=0.3),
                            nn.Linear(512,4))
    
    if args.channels_last:
        model = model.to(args.device, memory_format=torch.channels_last)
    else:
        model = model.to(args.device)
        
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    return model