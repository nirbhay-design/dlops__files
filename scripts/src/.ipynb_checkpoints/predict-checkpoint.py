import matplotlib.pyplot as plt
import seaborn as sns, os
from sklearn.metrics import confusion_matrix
from conf import  *
from utils import *
from model import *
from data import prepare_dataloader
import pandas as pd
import numpy as np, pickle
from PIL import Image
from scipy.special import softmax
import time

if args.mode == 'tensorrt':
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
if args.mode == 'onnx':
    import onnx
    import onnxruntime
    import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU_ID


def pytorch_infer():
    model = build_model()
    model.load_state_dict(torch.load(args.out_weight))
    if args.FP16:
        model = model.half()

    test_dl = prepare_dataloader()
    #Warmup
    with torch.no_grad():
        for i, (img, gt) in enumerate(test_dl):
            img, gt = img.to(args.device), gt.to(args.device)
            if args.FP16:
                img= img.half()
            out = model(img)
            if i==5:
                break
    test_dl = prepare_dataloader()
    data_pytorch_model = eval(model, test_dl)
    return data_pytorch_model['gt'], data_pytorch_model['pred']

def onnx_infer():
    session =onnxruntime.InferenceSession(args.triton_dir + args.model_name + '_onnx/1/model.onnx')
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    dataloader, gt = prepare_dataloader()
    pred = None
    #Warmup
    for i, val_imgs in enumerate(dataloader):
        data = json.dumps({'data': val_imgs.tolist()})
        data = np.array(json.loads(data)['data']).astype('float32')
        result = session.run([output_name], {input_name: data})
        if i==5:
            break
    dataloader, gt = prepare_dataloader()
    elapsed_time = []
    for i, val_imgs in enumerate(dataloader):
        data = json.dumps({'data': val_imgs.tolist()})
        data = np.array(json.loads(data)['data']).astype('float32')
        start_time = time.time()
        result = session.run([output_name], {input_name: data})
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        if i % 10 == 0:
            print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-10:].mean()) * 1000))
        result = np.argmax(softmax(result[0], axis=1), axis=1)
        pred = np.concatenate((pred, result), axis=0) if pred is not None else result
    pred = pred[:len(gt)]
    print('Throughput: {:.0f} images/s'.format(i * args.batch_size / elapsed_time.sum()))
    return gt, pred

def trt_infer():
    if args.TRTFP16:
        trt_engine = TensorRTInfer(args.triton_dir + args.model_name + '_trt_fp16/1/model.plan', args.batch_size)
    else:
        trt_engine = TensorRTInfer(args.triton_dir + args.model_name + '_trt_fp32/1/model.plan', args.batch_size)
        
    dataloader, gt = prepare_dataloader()
    pred = None
    #Warmup
    for i, val_imgs in enumerate(dataloader):
        result = trt_engine.infer(val_imgs)
        if i==5:
            break
    dataloader, gt = prepare_dataloader()
    elapsed_time = []
    for i, val_imgs in enumerate(dataloader):
        start_time = time.time()
        result = trt_engine.infer(val_imgs)
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        if i % 10 == 0:
            print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-10:].mean()) * 1000))
        result = np.argmax(softmax(result, axis=1), axis=1)
        pred = np.concatenate((pred, result), axis=0) if pred is not None else result
    print('Throughput: {:.0f} images/s'.format(i * args.batch_size / elapsed_time.sum()))
    pred = pred[:len(gt)]
    return gt, pred

def infer(mode = args.mode):
    if mode == 'pytorch':
        return pytorch_infer()
    elif mode == 'onnx':
        return onnx_infer()
    elif mode == 'tensorrt':
        return trt_infer()
    
def main():
    plt.figure(figsize = (10,7))
    gt, pred = infer()
    pickle.dump(pred, open('{}/{}_pred.pkl'.format('/'.join(args.out_weight.split('/')[:-1]), args.mode), 'wb'))
    fig_ = sns.heatmap(confusion_matrix(gt,pred), 
                       annot=True, cmap='Spectral', fmt='g').get_figure()
    
    plt.savefig('{}/{}_confusion_matrix.jpg'.format('/'.join(args.out_weight.split('/')[:-1]), args.mode))
    
if __name__ == '__main__':
    main()