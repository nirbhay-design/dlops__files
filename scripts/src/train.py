import os
import time
import random
import numpy as np
import pandas as pd
from data import prepare_dataloader
from model import *
from utils import *
from conf import *
if args.wandb:
    import wandb
    
if args.amp:
    from apex import amp
    
api_key = '5c52d5cd9234249d52721ffae2f9da0070886c6b'
os.environ['WANDB_API_KEY'] = api_key
    
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU_ID

def train_epoch(model, criterion, optimizer, train_dl, epoch, batch_ct = 0, example_ct = 0):
    if args.wandb:
        wandb.watch(model, criterion, log="all", log_freq=10)
    ############################################################                                          
    print('Epoch {}/{}'.format(epoch, args.epochs - 1))
    print('-' * 10)

    #Training
    model.train()
    for x,y in train_dl: #BS=32 ([BS,3,224,224], [BS,4]) 
        if args.channels_last:
            x = x.to(args.device, memory_format=torch.channels_last) #CHW --> #HWC
        else:
            x = x.to(args.device)
        y = y.to(args.device) 

        optimizer.zero_grad()

        train_logits = model(x) 
        _, train_preds = torch.max(train_logits, 1)
        train_loss = criterion(train_logits,y)

        if args.amp:
            with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                loss=scaled_loss
        ########################################################################
        else:
            train_loss.backward() # Backpropagation this is where your W_gradient
            loss=train_loss

        optimizer.step() # W_new = W_old - LR * W_gradient 
        example_ct += len(x) 
        batch_ct += 1
        
        if ((batch_ct + 1) % 25) == 0:
            train_log(loss, example_ct, epoch)
        ########################################################################
    return batch_ct, example_ct

def valid_epoch(model, criterion, optimizer, valid_dl, valid_ds):
    #validation
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    # Disable gradient calculation for validation or inference using torch.no_rad()
    with torch.no_grad():
        for x,y in valid_dl:
            if args.channels_last:
                x = x.to(args.device, memory_format=torch.channels_last) #CHW --> #HWC
            else:
                x = x.to(args.device)
            y = y.to(args.device) #CHW --> #HWC
            valid_logits = model(x)
            _, valid_preds = torch.max(valid_logits, 1)
            valid_loss = criterion(valid_logits,y)
            running_loss += valid_loss.item() * x.size(0)
            running_corrects += torch.sum(valid_preds == y.data)
            total += y.size(0)
            ########################################################################
            # Test Accuracy Logs
            if args.wandb:
                wandb.log({"test_accuracy": running_corrects / total})
            ########################################################################

    epoch_loss = running_loss / len(valid_ds)
    epoch_acc = running_corrects.double() / len(valid_ds)
    print("Validation Loss is {}".format(epoch_loss))
    print("Validation Accuracy is {}".format(epoch_acc.cpu()))
    return epoch_acc.cpu()
    
def train(model, criterion, optimizer, train_dl, valid_dl, valid_ds):
    since = time.time() 
    batch_ct, example_ct = 0, 0
    best_accuracy = 0.0
    for epoch in range(args.epochs):
        batch_ct, example_ct = train_epoch(model, criterion, optimizer, train_dl, epoch, batch_ct, example_ct)
        accuracy = valid_epoch(model, criterion, optimizer, valid_dl, valid_ds)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), args.out_weight)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    torch.save(model.state_dict(), args.out_weight.replace(f'{args.model_name}', f'{args.model_name}_last'))

def main():
    if args.wandb:
        wandb.init(project=args.project_name, config=args)
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
    if args.seed:
        seeding(args.seed)
    if args.benchmark:
        benchmark()
    if args.distributed:
        distributed()
    _, valid_ds, train_dl, valid_dl = prepare_dataloader()
    model = build_model()
    model, optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss()
    train(model, criterion, optimizer, train_dl, valid_dl, valid_ds)

if __name__ == '__main__':
    main()