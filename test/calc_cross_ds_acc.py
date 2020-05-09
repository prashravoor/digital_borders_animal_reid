import os
import sys
sys.path.insert(0, '../reid-strong-baseline')
from modeling import Baseline
from calc_closed_reid_acc import getAverageCmcMap

if __name__ == '__main__':
    args = sys.argv
    if not len(args) == 11:
        print('Usage: <5 models> <5 image folders>')
        exit(1)
        
    models = args[1:6]
    folders = args[6:]

    print('Models: {}'.format(models))
    print('Folders: {}'.format(folders))
    all_results = []
    for m in models:
        model_res = []
        for f in folders:
            print('Measuring performance for model {} over folder: {}'.format(m, f))
            mAP, means, stds = getAverageCmcMap(m, f)
            model_res.append('"mAP: {:.4f}, Mean Accuracy: {}"'.format(mAP, '{:.4f} +- {:.4f}'.format(means[0], stds[0])))
            
        all_results.append(model_res)
        
    print(','.join([x for x in folders]))
    for i in range(len(models)):
        print('{},{}'.format(models[i], ','.join(all_results[i])))    