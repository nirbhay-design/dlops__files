from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from apex.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from conf import *
import os
import json

class PlantPathologyDataset(Dataset):
    def __init__(self,x,y,ds_type):
        self.x = x # File Path in CSV
        self.y = y # Label in CSV
        self.ds_type = ds_type
        self.transforms = self._build_augmentations()
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx): #File Name --> Preprocessed 3-D Tensor
        fname = self.x.iloc[idx]        
        label = self.y.iloc[idx]
        image = Image.open(fname)
        
        if self.transforms:
            image = self.transforms(image)

        return image, label #[3,224,224], [0-3] 
    def _build_augmentations(self,):
        augmentations = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop((args.image_size,args.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((args.image_size,args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        return augmentations[self.ds_type]
    
trans = transforms.Compose([
                transforms.Resize((args.image_size,args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
def create_fname(path,extension):
    def add_extension(fname):
        return os.path.join(path,fname)+extension
    return add_extension
    
def data_manipulation(path, mode = args.mode):
    train_df = pd.read_csv(path)
    vocab = json.load(open(args.vocab))
    if mode == 'train':
        test_df  = pd.read_csv(args.df_test)
        df_fnames = train_df["image_id"].append(test_df["image_id"],ignore_index=True).tolist()
    else:
        return train_df, vocab
        
    jpeg_extension_creator = create_fname(args.image_dir,".jpg")
    train_df["image_id"] = train_df["image_id"].apply(jpeg_extension_creator)
    if mode == 'train':
        test_df["image_id"] = test_df["image_id"].apply(jpeg_extension_creator)
        
    for label in vocab:
        train_df.loc[train_df[label] == 1, "label" ] = vocab[label] 
    train_df["label"] = train_df["label"].astype(int)
    return train_df, vocab

def data_splitting(train_df):
    train_df_X, valid_df_X, train_df_y, valid_df_y = train_test_split(train_df["image_id"],
                                                                  train_df["label"], 
                                                                  test_size=args.test_size, 
                                                                  random_state=0)
    train_df_split = pd.DataFrame(data={"image_id": train_df_X, "label": train_df_y})
    train_df_split.to_csv(os.path.join(args.data_dir, "train_split.csv"), index=False)

    valid_df_split = pd.DataFrame(data={"image_id": valid_df_X, "label": valid_df_y})
    valid_df_split.to_csv(os.path.join(args.data_dir, "val_split.csv"), index=False)
    """
    print("Number of train input samples is {}".format(len(train_df_X)))
    print("Number of valid input samples is {}".format(len(valid_df_X)))
    print("Number of train output samples is {}".format(len(train_df_y)))
    print("Number of valid output samples is {}".format(len(valid_df_y)))
    """
    return train_df_X, valid_df_X, train_df_y, valid_df_y
    
    
def prepare_train_dataloader():
    train_df, vocab = data_manipulation(args.df_train)
    train_df_X, valid_df_X, train_df_y, valid_df_y = data_splitting(train_df)
    
    train_ds = PlantPathologyDataset(train_df_X, train_df_y, 'train',)
    valid_ds = PlantPathologyDataset(valid_df_X, valid_df_y, 'val',)
    
    if args.distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=args.world_size,
            rank=args.local_rank
        )
        
        train_dl = DataLoader(train_ds,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.num_workers,
                      pin_memory=args.pin_memory,
                      sampler=train_sampler)
    else:
        train_dl = DataLoader(train_ds,
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=args.num_workers,
                      pin_memory=args.pin_memory)
        
    valid_dl = DataLoader(valid_ds,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.num_workers,
                      pin_memory=args.pin_memory)
    
    return train_ds, valid_ds, train_dl, valid_dl

def prepare_test_dataloader():
    test_df, vocab = data_manipulation(args.df_val)
    test_ds = PlantPathologyDataset(test_df["image_id"], test_df['label'], 'val',)
    return DataLoader(test_ds, batch_size=args.batch_size)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield np.array( [trans(Image.open(path)).numpy() for path in iterable[ndx:min(ndx + n, l)]] , dtype=np.float32)
        
def prepare_nonpytorch_data():
    val_split = pd.read_csv(args.df_val)
    val_img_paths = [path for path in val_split['image_id']]
    return batch(val_img_paths, args.batch_size), val_split['label']

def prepare_dataloader(mode = args.mode):
    if mode == 'train':
        return prepare_train_dataloader()
    elif mode =='pytorch':
        return prepare_test_dataloader()
    else:
        return prepare_nonpytorch_data()
    
    
    

