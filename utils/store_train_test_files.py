import sys
import os

if __name__ == '__main__':
    args = sys.argv
    if not len(args) == 3:
        print('Usage: cmd <train folder> <test folder>')
        exit()
        
    trainfolder = args[1]
    testfolder = args[2]
    
    metafilename = 'class_mapping.txt'
    if not os.path.exists(os.path.join(trainfolder, metafilename)):
        print('Train folder does not contain the mapping file: {}'.format(metafilename))
        exit()
        
    if not os.path.exists(os.path.join(testfolder, metafilename)):
        print('Test folder does not contain the mapping file: {}'.format(metafilename))
        exit()
    

    def loadMapping(folder, metafile):
        with open(os.path.join(folder, metafile)) as f:
            mapping = {x.split()[0].strip() : x.split()[1].strip() for x in f.readlines()}
            
        return mapping
        
    train_mapping = loadMapping(trainfolder, metafilename)
    test_mapping = loadMapping(testfolder, metafilename)
    
    trainfiles = [x for x in os.listdir(trainfolder) if x.endswith('.jpg') or x.endswith('.png') and x in train_mapping]
    testfiles = [x for x in os.listdir(testfolder) if x.endswith('.jpg') or x.endswith('.png') and x in test_mapping]
    
    def remap_files(files, folder, metafilename='file_map.txt'):
        if os.path.exists(os.path.join(folder, metafilename)):
            with open(os.path.join(folder, metafilename)) as f:
                mapping = {x.split('\t')[1].strip() : x.split('\t')[0].strip() for x in f.readlines()}
            files = [mapping[x] for x in files]
        return files
    
    trainfiles = remap_files(trainfiles, trainfolder)
    testfiles = remap_files(testfiles, testfolder)
    
    trainfile = os.path.join(trainfolder, 'train_file_list.txt')
    testfile = os.path.join(testfolder, 'test_file_list.txt')
    
    with open(trainfile, 'w') as f:
        f.write('\n'.join(trainfiles))
        
    with open(testfile, 'w') as f:
        f.write('\n'.join(testfiles))