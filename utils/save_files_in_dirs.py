import sys
import os

if __name__ == '__main__':
    args = sys.argv
    if not len(args) == 3:
        print('Usage: cmd <in folder> <mapping file>')
        exit()
        
    folder = args[1]
    metafile = args[2]
    
    if not os.path.exists(metafile):
        print('Mapping file not found: {}'.format(metafile))
        exit()

    def loadMapping(metafile):
        with open(metafile) as f:
            mapping = {x.split('\t')[0].strip() : x.split('\t')[1].strip() for x in f.readlines()}

        return mapping

    file_mapping = loadMapping(metafile)

    folders = [os.path.join(folder, 'image_gallery'), os.path.join(folder, 'image_train'), os.path.join(folder, 'image_query')]
    for dir in folders:
        files = [file_mapping[x] for x in os.listdir(dir) if x.endswith('.jpg') or x.endswith('.png')]
        outfilename = os.path.join(folder, '{}.txt'.format(os.path.basename(dir)))
        with open(outfilename, 'w') as f:
            f.write('\n'.join(files))
