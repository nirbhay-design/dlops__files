from datetime import datetime
import os

abs_path = os.path.dirname(__file__)
now = datetime.now()
now = now.strftime("%d_%m_%Y")

args = {
    'data_dir'                 :'/workspace/Pytorch_CV_Lab/data',
    'image_dir'                :'/workspace/Pytorch_CV_Lab/data/images',
    'df_train'                 :'/workspace/Pytorch_CV_Lab/data/train.csv',
    'df_val'                   :'/workspace/Pytorch_CV_Lab/data/val_split.csv',
    'df_test'                  :'/workspace/Pytorch_CV_Lab/data/test.csv',
    'vocab'                    :'/workspace/Pytorch_CV_Lab/scripts/src/labels.json',
    'triton_dir'               :"/workspace/Pytorch_CV_Lab/triton_model_repository/",

    'GPU_ID'                   : '0',
    'batch_size'               : 32,
    'num_workers'              : 4,
    'pin_memory'               : True,
    'channels_last'            : False,
    'epochs'                   : 10,
    'test_size'                : 0.2,
    'learning_rate'            : 0.001,
    

    ######### MODEL parameters ########
    'model_name'               :'resnet50',
    'out_weight_dir'           :'/workspace/Pytorch_CV_Lab/saved',
    'out_weight'               :'/workspace/Pytorch_CV_Lab/saved/resnet50.pt',
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
    'TF32'                     : True, 
    'benchmark'                : True,
}
