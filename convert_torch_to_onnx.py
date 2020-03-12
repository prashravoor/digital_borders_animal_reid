import torch
import onnx
import sys
import onnx

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
torch.onnx.export(model, dummy, outmodel, keep_initializers_as_inputs=True, export_params=True)
