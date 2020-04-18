import json
import os
import shutil
import sys
import numpy as np

args = sys.argv
if len(args) < 3:
    print('Usage: cmd <in folder> <out folder> <split number>')
    exit()
    
base = args[1]
outfolder = args[2]
if len(args) >= 4:
    split = int(args[3])
else:
    split = np.random.randint(0, 10)
    
splitsfile = os.path.join(base, 'splits.json')
metafile = os.path.join(base, 'meta.json')
imagesfolder = os.path.join(base, 'images')

with open(metafile) as f:
    metadata = json.load(f)

with open(splitsfile) as f:
    splits = json.load(f)

identities = metadata['identities']
num_cams = metadata['num_cameras']

print('Total Identities: {}, Num cams: {}, using split id {} to partition the data'.format(len(identities), num_cams, split))

def makeDir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    
trainf = os.path.join(outfolder, 'image_train')
queryf = os.path.join(outfolder, 'image_query')
galf = os.path.join(outfolder, 'image_gallery')

makeDir(outfolder)
makeDir(trainf)
makeDir(queryf)
makeDir(galf)

train_pids = splits[split]['trainval']
query_pids = splits[split]['query']
gallery_pids = splits[split]['gallery']

print('Total Train Ids: {}, Query Ids: {}, Gallery Ids: {}'.format(len(train_pids), len(query_pids), len(gallery_pids)))

def copyFilesFromSplit(identities, pids, num_cams, infolder, outfolder):
    counter = 0
    for pid in pids:
        for i in range(num_cams):
            for img in identities[pid][i]:
                shutil.copyfile(os.path.join(infolder, img), os.path.join(outfolder, img))
                counter += 1
    print('Copied {} files into {} set'.format(counter, outfolder))

copyFilesFromSplit(identities, train_pids, num_cams, imagesfolder, trainf)
copyFilesFromSplit(identities, query_pids, num_cams, imagesfolder, queryf)
copyFilesFromSplit(identities, gallery_pids, num_cams, imagesfolder, galf)
