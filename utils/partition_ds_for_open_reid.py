import json
import os
import shutil
import sys
import numpy as np

def copyPartition(base, outfolder, split):
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

    # For query and gallery set, split them 30% to query and rest to gallery
    assert sorted(query_pids) == sorted(gallery_pids) # Closed set

    qcounter = 0
    gcounter = 0
    for pid in query_pids:
        # single camera only. for multiple cams add support here later
        images = identities[pid][0]
        if len(images) <= 1:
            print('Skipping identity {}, too few images: {}'.format(pid, len(images)))
            continue

        num = int(np.ceil(0.3 * len(images)))
        query = np.random.choice(images, num, replace=False)
        gallery = [x for x in images if not x in query]
        
        qcounter += len(query)
        gcounter += len(gallery)
        for img in query:
            shutil.copyfile(os.path.join(imagesfolder, img), os.path.join(queryf, img))
        
        for img in gallery:
            shutil.copyfile(os.path.join(imagesfolder, img), os.path.join(galf, img))
            
    print('Copied {} files into Query folder, and {} files into gallery folder'.format(qcounter, gcounter))

if __name__ == '__main__':
    args = sys.argv
    if len(args) < 3:
        print('Usage: cmd <in folder> <out folder> [split number]')
        exit()

    np.random.seed(42)

    base = args[1]
    outfolder = args[2]
    if len(args) >= 4:
        split = int(args[3])
    else:
        split = np.random.randint(0, 10)
        
    copyPartition(base, outfolder, split)