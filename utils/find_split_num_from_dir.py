import os
import sys
import json

if __name__ == '__main__':
    args = sys.argv
    if len(args) < 3:
        print('Usage: cmd <reid-data folder> <splits.json> file')
        exit()
        
    folder = args[1]
    metafile = args[2]
    
    qry = os.path.join(folder, 'image_query')
    trn = os.path.join(folder, 'image_train')
    
    with open(metafile) as f:
        splits = json.load(f)

    qry_ids = set([int(x.split('_')[0]) for x in os.listdir(qry) if x.endswith('.jpg')])
    trn_ids = set([int(x.split('_')[0]) for x in os.listdir(trn) if x.endswith('.jpg')])

    for i in range(len(splits)):
        split = splits[i]
        if set(split['trainval']) == trn_ids:
            if set(split['query']) == qry_ids:
                print('Found matching Split: {}'.format(i))
                exit()
                
    print('No matching split found! -------xx-----------')
    