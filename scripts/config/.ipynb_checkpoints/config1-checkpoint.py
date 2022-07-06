from datetime import datetime
import os

abs_path = os.path.dirname(__file__)
now = datetime.now()
now = now.strftime("%d_%m_%Y")

args = {
    'data_dir'                 :'/Pytorch_CV_Lab/data',
    'image_dir'                :'/Pytorch_CV_Lab/data/images',
    'df_train'                 :'/Pytorch_CV_Lab/data/train.csv',
    'df_val'                   :'/Pytorch_CV_Lab/data/val_split.csv',
    'df_test'                  :'/Pytorch_CV_Lab/data/test.csv',
    'vocab'                    :'/Pytorch_CV_Lab/scripts/src/labels.json',
    'triton_dir'               :"/workspace/triton_model_repository/",

    'GPU_ID'                   : '0',
    'batch_size'               : 16,
    'num_workers'              : 4,
    'pin_memory'               : True,
    'channels_last'            : False,
    'epochs'                   : 10,
    'test_size'                : 0.2,
    'learning_rate'            : 0.001,
    

    ######### MODEL parameters ########
    'model_name'               :'resnet18',
    'out_weight_dir'           :'/workspace/saved',
    'out_weight'               :'/workspace/saved/resnet18.pt',
    'image_size'               : 224,
    ##################################

    'distributed'              : False,
    'local_rank'               : 0,
    'amp'                      : True,
    'opt_level'                : "O2",
    'wandb'                    : True,
    'project_name'             : 'iitj_lecture3',
    'device'                   : 'gpu',
    'seed'                     : 42,
    'world_size'               : 4,
    'TF32'                     : False, 
    'FP16'                     : True,
    'TRTFP16'                  : True,
    'benchmark'                : True,
}