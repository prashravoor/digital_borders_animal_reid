import torch
import sys
from pytorch2keras import pytorch_to_keras
import tensorflow as tf

args = sys.argv
if not len(args) == 3:
    print('Usage: cmd <input model> <output path>')
    exit()

inmodel = args[1]
outmodel = args[2]

sys.path.insert(0, '../reid-strong-baseline')
model = torch.load(inmodel, map_location=torch.device('cpu'))
model.train(False)

dummy = torch.randn(1, 3, 256, 256)
#torch.onnx.export(model, dummy, outmodel, keep_initializers_as_inputs=True, export_params=True)
# we should specify shape of the input tensor
k_model = pytorch_to_keras(model, dummy, verbose=True)
tf.saved_model.save(k_model, outmodel)
