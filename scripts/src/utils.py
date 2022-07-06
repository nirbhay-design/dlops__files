from conf import  *
import random, time, numpy as np, torch, torch.nn as nn, torch.optim as optim
if args.wandb:
    import wandb

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

if args.amp:
    from apex import amp
    from apex.optimizers import FusedAdam

class TensorRTInfer:
    """
    Implements inference for the EfficientNet TensorRT engine.
    """

    def __init__(self, engine_path, batch_size):
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context
        
        self.context.set_binding_shape(0, (args.batch_size, 3, 224 , 224))
        self.batch_size = batch_size

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            
            if (shape[0] < 0):
                shape[0] = self.batch_size
                
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
                
        # Prepare the output data
        self.output_data = np.zeros(*self.output_spec())

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        return self.outputs[0]['shape'], self.outputs[0]['dtype']

    def infer(self, batch, top=1):       
        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        cuda.memcpy_dtoh(self.output_data, self.outputs[0]['allocation'])
        return self.output_data

def eval (model, dataloader):    
    model.eval()
    gt_all = None
    pred_all = None
    elapsed_time = []
    with torch.no_grad():
        for i, (img, gt) in enumerate(dataloader):

            img, gt = img.to(args.device), gt.to(args.device)

            start_time = time.time()
            out = model(img).cpu()
            end_time = time.time()
            elapsed_time = np.append(elapsed_time, end_time - start_time)
            if i % 10 == 0 and i != 0:
                print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-10:].mean()) * 1000))


            out = torch.softmax(out.float(), dim=1).detach().cpu()

            gt_all = np.concatenate((gt_all, gt.detach().cpu()), axis=0) if gt_all is not None else gt.detach().cpu()
            pred_all = np.concatenate((pred_all, out), axis=0) if pred_all is not None else out
    print('Throughput: {:.0f} images/s'.format(i * args.batch_size / elapsed_time.sum()))
    return {
        "gt" : gt_all ,
        "pred" : np.argmax(pred_all, axis=1)
    }

def train_log(loss, example_ct, epoch):
    loss = float(loss)
    # where the magic happens
    if args.wandb:
        wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    
def seeding(seed):
    # For custom operators, you might need to set python seed
    random.seed(args.seed)
    # If you or any of the libraries you are using rely on NumPy, you can seed the global NumPy RNG 
    np.random.seed(args.seed)
    # Prevent RNG for CPU and GPU using torch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
def benchmark():
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    ## Enabling TF32 on Ampere GPUs
    if args.TF32:
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
        torch.backends.cuda.matmul.allow_tf32 = True
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True
    
def distributed():
    # FOR DISTRIBUTED:  Set the device according to local_rank.
    torch.cuda.set_device(args.local_rank)       
    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank = args.local_rank, world_size=args.world_size)

def build_optimizer(model):
    if args.amp:
        optimizer = FusedAdam(model.parameters(), args.learning_rate)
        model,optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level) 
    else:
        optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    return model, optimizer
