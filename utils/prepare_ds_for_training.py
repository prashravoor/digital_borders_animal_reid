import os
import sys
import numpy as np
import random
import shutil
from glob import glob
import os.path as osp
import json
from partition_ds_for_open_reid import copyPartition

def write_json(obj, fpath):
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

# Borrowed most of the code from `open-reid` GitHub Repository
def generateSplitsForData(raw_dir, outfolder, dsName, metafilename='class_mapping.txt'):
    if not osp.exists(raw_dir):
        print('Folder {} does not exist'.format(raw_dir))
        exit()
    
    if not osp.exists(outfolder):
        os.mkdir(outfolder)
    
    images_dir = osp.join(outfolder, 'images')
    # Format
    if not osp.exists(images_dir):
        os.mkdir(images_dir)

    with open(osp.join(raw_dir, metafilename)) as f:
        file_map = {x.split()[0].strip() : x.split()[1].strip() for x in f.readlines()}

    rev_map = dict()
    cam = 0
    for k,v in file_map.items():
        if not v in rev_map:
            rev_map[v] = []
        rev_map[v].append(k)

    # Account for missing identity numbers, relabel each one to get continuous identities
    counter = 0
    id_map = dict()
    for k in rev_map.keys():
        id_map[k] = counter
        counter += 1

    num_identities = len(rev_map)
    num_cameras = 1 # Always a single camera. No support for multi-camera input data
    
    identities = [[[] for _ in range(num_cameras)] for _ in range(num_identities)]

    files = sorted(glob(osp.join(raw_dir, '*.jpg'))) # Default images are jpg
    if len(files) == 0:
        # Or Png
        files = sorted(glob(osp.join(raw_dir, '*.png')))
    
    if len(files) == 0:
        print('No images in the dataset')
        exit()

    filename_map = dict()
    for fpath in files:
        fname = osp.basename(fpath)
        pid, cam = int(id_map[file_map[fname]]) + 1, 1
        assert 1 <= pid <= num_identities
        assert 1 <= cam <= num_cameras
        pid, cam = pid - 1, (cam - 1) // 2
        fname = ('{:08d}_{:02d}_{:04d}.jpg' # Same format is required by the code in reid-strong-baseline
                 .format(pid, cam, len(identities[pid][cam])))
        identities[pid][cam].append(fname)
        shutil.copy(fpath, osp.join(images_dir, fname))
        filename_map[fname] = osp.basename(fpath)
        
    with open(os.path.join('filename_mapping.txt'), 'w') as f:
        f.write('\n'.join(['{}\t{}'.format(k,v) for k,v in filename_map.items()]))

    # Save meta information into a json file
    meta = {'name': dsName, 'shot': 'multiple', 'num_cameras': 1,
            'identities': identities}
    write_json(meta, osp.join(outfolder, 'meta.json'))

    # Randomly create ten training and test split
    num = len(identities)
    splits = []
    for _ in range(10):
        pids = np.random.permutation(num).tolist()
        
        # Hack time! Needs to be handled a lot better...
        # For jaguars and elephants, use 75% of the dataset for training. Else use 50%
        if dsName.lower() in ['elp', 'jaguar']:
            test_pids = sorted(pids[: num // 4])
            trainval_pids = sorted(pids[num // 4:])
        else:
            trainval_pids = sorted(pids[:num // 2])
            test_pids = sorted(pids[num // 2:])

        split = {'trainval': trainval_pids,
                 'query': test_pids,
                 'gallery': test_pids}
        splits.append(split)
    write_json(splits, osp.join(outfolder, 'splits.json'))

if __name__ == '__main__':
    args = sys.argv
    if len(args) < 5 or len(args) > 6:
        print('Usage: cmd <Input train images folder> <Intermediate output folder> <Output folder path> <DS Name> [partition id]')
        exit()
        
    infolder = args[1]
    int_outfolder = args[2]
    outfolder = args[3]
    dsName = args[4]
    
    np.random.seed(42)
    random.seed(42)
    
    DS_NAMES = ['amur', 'elp', 'jaguar', 'chimp', 'facescrub']
    if not dsName in DS_NAMES:
        print('DS Name has to be one of {}'.format(DS_NAMES))
        exit()
    
    partition = 0
    if len(args) > 5:
        try:
            partition = int(args[5])
        except:
            pass

    print('Creating Dataset for {}, using Partition Id: {}'.format(dsName, partition))
    print('Creating intermediate output and metadata...')
    generateSplitsForData(infolder, int_outfolder, dsName)
    
    if not osp.exists(outfolder):
        os.mkdir(outfolder)
    
    print('Completed creating metadata. Creating output folder using partition {}'.format(partition))
    copyPartition(int_outfolder, outfolder, partition)
    