from conf import *
from model import *
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import os

NUM_CLASSES = 4
INPUT_SHAPE = (3, 224, 224)

def main():
    os.system(f'rm -rf {args.triton_dir}')

    os.makedirs(f"{args.triton_dir}{args.model_name}_torch/1")
    os.makedirs(f"{args.triton_dir}{args.model_name}_onnx/1")
    os.makedirs(f"{args.triton_dir}{args.model_name}_trt_fp32/1")
    os.makedirs(f"{args.triton_dir}{args.model_name}_trt_fp16/1")
    os.makedirs(f"{args.triton_dir}{args.model_name}_trt_int8/1")
    
    JIT_MODEL_PATH = f'{args.triton_dir}{args.model_name}_torch/1/model.pt'
    ONNX_MODEL_PATH = f'{args.triton_dir}{args.model_name}_onnx/1/model.onnx'
    TRT_MODEL_PATH = f'{args.triton_dir}{args.model_name}_trt_fp32/1/model.plan'
    TRT_MODEL_PATH_FP16 = f'{args.triton_dir}{args.model_name}_trt_fp16/1/model.plan'
    TRT_MODEL_PATH_INT8 = f'{args.triton_dir}{args.model_name}_trt_int8/1/model.plan'
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU_ID

    model = build_model()
    model.load_state_dict(torch.load(args.out_weight))
    model.eval()

    if args.channels_last:
        example = torch.randn((args.batch_size, *INPUT_SHAPE), dtype=torch.float32, device=args.device).to(memory_format=torch.channels_last)
    else:
        example = torch.randn((args.batch_size, *INPUT_SHAPE), dtype=torch.float32, device=args.device)

    script = torch.jit.trace(model, example)
    script.save(JIT_MODEL_PATH)


    if args.channels_last:
        x = torch.randn((1, *INPUT_SHAPE), dtype=torch.float32, device=args.device).to(memory_format=torch.channels_last)
    else:
        x = torch.randn((1, *INPUT_SHAPE), dtype=torch.float32, device=args.device)

    torch.onnx.export(model,                       # model being run
                      x,                           # model input (or a tuple for multiple inputs)
                      ONNX_MODEL_PATH,             # Path to saved onnx model
                      export_params=True,          # store the trained parameter weights inside the model file
                      opset_version=13,            # the ONNX version to export the model to
                      input_names = ['input'],     # the model's input names
                      output_names = ['output'],   # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    os.system(f'trtexec --onnx={ONNX_MODEL_PATH} --explicitBatch --workspace=40000 --optShapes=input:8x3x224x224 --maxShapes=input:128x3x224x224 --minShapes=input:1x3x224x224 --saveEngine={TRT_MODEL_PATH}')

    os.system(f'trtexec --onnx={ONNX_MODEL_PATH} --explicitBatch --workspace=40000 --optShapes=input:8x3x224x224 --maxShapes=input:128x3x224x224 --minShapes=input:1x3x224x224 --saveEngine={TRT_MODEL_PATH_FP16} --fp16')
    
    os.system(f'trtexec --onnx={ONNX_MODEL_PATH} --explicitBatch --workspace=40000 --optShapes=input:8x3x224x224 --maxShapes=input:128x3x224x224 --minShapes=input:1x3x224x224 --saveEngine={TRT_MODEL_PATH_INT8} --int8')
if __name__ == '__main__':
    main()
