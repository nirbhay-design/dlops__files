import sys, os
import importlib
from types import SimpleNamespace
import argparse, torch

sys.path.append("/Pytorch_CV_Lab/scripts/config")

parser = argparse.ArgumentParser(description='')

parser.add_argument("-C", "--config", default="config1", help="config filename")
parser.add_argument("--mode", default = 'train', type = str, help = "Choose any of these {train, convert, pytorch, onnx, tensorrt}")
parser.add_argument("--distributed", help="Running DDP?", action="store_true")

parser_args, _ = parser.parse_known_args(sys.argv)
# parser_args = parser.parse_args()

print("Using config file", parser_args.config)

args = importlib.import_module(parser_args.config).args

args['mode'] = parser_args.mode
args['distributed'] = parser_args.distributed

if args['device'] == 'cpu':
    args['device'] = torch.device('cpu')
else:
    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
args =  SimpleNamespace(**args)
os.makedirs(args.out_weight_dir, exist_ok = True)
