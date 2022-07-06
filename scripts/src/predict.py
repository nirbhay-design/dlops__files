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
import torch
from torchvision import transforms
if args.mode == 'tensorrt32':
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

if args.mode == 'tensorrt16':
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit


if args.mode == 'tensorrt8':
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

if args.mode == 'onnx':
    import onnx
    import onnxruntime
    import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU_ID

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def pytorch_infer(val_imgs):
    model = build_model()
    model.load_state_dict(torch.load(args.out_weight))

    #Warmup    
    input_batch_gpu = torch.from_numpy(val_imgs).to(args.device)
    with torch.no_grad():
        preds = np.array(model(input_batch_gpu).cpu())
    
    #Inference
    test_dl = prepare_dataloader()
    data_pytorch_model = eval(model, test_dl)
    return data_pytorch_model['gt'], data_pytorch_model['pred']

def onnx_infer(val_imgs):
    session =onnxruntime.InferenceSession(args.triton_dir + args.model_name + '_onnx/1/model.onnx')
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    #Warmup
    data0 = json.dumps({'data': val_imgs.tolist()})
    data0 = np.array(json.loads(data0)['data']).astype('float32')
    preds = session.run([output_name], {input_name: data0})

    #Inference
    dataloader, gt = prepare_dataloader()
    pred = None
    
    elapsed_time = []
    for i, val_imgs in enumerate(dataloader):
        data = json.dumps({'data': val_imgs.tolist()})
        data = np.array(json.loads(data)['data']).astype('float32')
        start_time = time.time()
        result = session.run([output_name], {input_name: data})
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        if i % 10 == 0 and i != 0:
            print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-10:].mean()) * 1000))
        result = np.argmax(softmax(result[0], axis=1), axis=1)
        pred = np.concatenate((pred, result), axis=0) if pred is not None else result
    pred = pred[:len(gt)]
    print('Throughput: {:.0f} images/s'.format(i * args.batch_size / elapsed_time.sum()))
    return gt, pred

def trt_infer(val_imgs):
    trt_engine = TensorRTInfer(args.triton_dir + args.model_name + '_trt_fp32/1/model.plan', args.batch_size)
        
    #Warmup
    preds = trt_engine.infer(val_imgs)

    #Inference
    dataloader, gt = prepare_dataloader()
    pred = None

    elapsed_time = []
    for i, val_imgs in enumerate(dataloader):
        start_time = time.time()
        result = trt_engine.infer(val_imgs)
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
    
        if i % 10 == 0 and i != 0:
            print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-10:].mean()) * 1000))
        result = np.argmax(softmax(result, axis=1), axis=1)
        pred = np.concatenate((pred, result), axis=0) if pred is not None else result
    print('Throughput: {:.0f} images/s'.format(i * args.batch_size / elapsed_time.sum()))
    pred = pred[:len(gt)]
    return gt, pred

def trt_infer16(val_imgs):
    trt_engine = TensorRTInfer(args.triton_dir + args.model_name + '_trt_fp16/1/model.plan', args.batch_size)
    
    #Warmup
    preds = trt_engine.infer(val_imgs)
    
    #Inference
    dataloader, gt = prepare_dataloader()
    pred = None

    elapsed_time = []
    for i, val_imgs in enumerate(dataloader):
        start_time = time.time()
        result = trt_engine.infer(val_imgs)
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        if i % 10 == 0 and i != 0:
            print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-10:].mean()) * 1000))
        result = np.argmax(softmax(result, axis=1), axis=1)
        pred = np.concatenate((pred, result), axis=0) if pred is not None else result
    print('Throughput: {:.0f} images/s'.format(i * args.batch_size / elapsed_time.sum()))
    pred = pred[:len(gt)]
    return gt, pred



def trt_infer8(val_imgs):
    trt_engine = TensorRTInfer(args.triton_dir + args.model_name + '_trt_int8/1/model.plan', args.batch_size)

    #Warmup
    preds = trt_engine.infer(val_imgs)

    #Inference
    dataloader, gt = prepare_dataloader()
    pred = None

    elapsed_time = []
    for i, val_imgs in enumerate(dataloader):
        start_time = time.time()
        result = trt_engine.infer(val_imgs)
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        if i % 10 == 0 and i != 0:
            print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-10:].mean()) * 1000))
        result = np.argmax(softmax(result, axis=1), axis=1)
        pred = np.concatenate((pred, result), axis=0) if pred is not None else result
    print('Throughput: {:.0f} images/s'.format(i * args.batch_size / elapsed_time.sum()))
    pred = pred[:len(gt)]
    return gt, pred



def infer(mode = args.mode):
    #warmup
    trans = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    val_split = pd.read_csv(args.df_val)
    val_img_paths = [ path for path in val_split['image_id'][:args.batch_size]]
    val_imgs = np.array([trans(Image.open(path)).numpy() for path in val_img_paths] , dtype=np.float32)

    if mode == 'pytorch':
        return pytorch_infer(val_imgs)
    elif mode == 'onnx':
        return onnx_infer(val_imgs)
    elif mode == 'tensorrt32':
        return trt_infer(val_imgs)
    elif mode == 'tensorrt16':
        return trt_infer16(val_imgs)
    elif mode == 'tensorrt8':
        return trt_infer8(val_imgs)
    
def main():
    plt.figure(figsize = (10,7))
    gt, pred = infer()
    pickle.dump(pred, open('{}/{}_pred.pkl'.format('/'.join(args.out_weight.split('/')[:-1]), args.mode), 'wb'))
    fig_ = sns.heatmap(confusion_matrix(gt,pred), 
                       annot=True, cmap='Spectral', fmt='g').get_figure()
    
    plt.savefig('{}/{}_confusion_matrix.jpg'.format('/'.join(args.out_weight.split('/')[:-1]), args.mode))
    
if __name__ == '__main__':
    main()
