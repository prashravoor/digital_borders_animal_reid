import os
import sys

args = sys.argv
if len(args) < 4:
    print('Usage: cmd <path to chimpanzee cropped faces> <intermediate output folder> <output folder> [partition number]')
    exit()

infolder = os.path.realpath(args[1])
int_outfolder = os.path.realpath(args[2])
outfolder = os.path.realpath(args[3])

split = 0
if len(args) > 4:
    try:
        split = int(args[4])
    except:
        pass

print('Using split {} to generate the dataset'.format(split))
        
ctai = os.path.join(infolder, 'data_CTai')
czoo = os.path.join(infolder, 'data_CZoo')

metafile_folder = os.path.join(os.path.dirname(infolder), 'demo_access_data')
czoo_identity_metafile = os.path.join(metafile_folder, 'dataset_splits_CZoo.mat')
ctai_identity_metafile = os.path.join(metafile_folder, 'dataset_splits_CTai.mat')

# Generate the metadata from the .mat file to a more readable form
# Generates two files under respective folders - split_<split num>_train.txt and split_<split num>_test.txt
if not os.system('python getDsSplits.py {} {} {}'.format(ctai, ctai_identity_metafile, split)) == 0:
    exit()
if not os.system('python getDsSplits.py {} {} {}'.format(czoo, czoo_identity_metafile, split)) == 0:
    exit()

# Create a temporary output folder, which simply combines czoo and ctai train and test folders respectively.
czoo_train_split = os.path.join(czoo, 'split_{}_train.txt'.format(split))
czoo_test_split = os.path.join(czoo, 'split_{}_test.txt'.format(split))
ctai_train_split = os.path.join(ctai, 'split_{}_train.txt'.format(split))
ctai_test_split = os.path.join(ctai, 'split_{}_test.txt'.format(split))

int_out_train = os.path.join(int_outfolder, 'train')
int_out_test = os.path.join(int_outfolder, 'test')
if not os.system('python combineTwoDs.py {} {} {}'.format(czoo_train_split, ctai_train_split, int_out_train)) == 0:
    exit()
if not os.system('python combineTwoDs.py {} {} {}'.format(czoo_test_split, ctai_test_split, int_out_test)) == 0:
    exit()

# Finally partition the train and test dirs by ID, instead of the default ordering
# Use same intermediate folder and delete the existing ones
out_train_byid = os.path.join(outfolder, 'train')
out_test_byid = os.path.join(outfolder, 'test')
mappingfile_train = os.path.join(int_out_train, 'class_mapping.txt')
mappingfile_test = os.path.join(int_out_test, 'class_mapping.txt')

os.system('python combineTwoDsById.py {} {} {} {}'.format(mappingfile_train, mappingfile_test, out_train_byid, out_test_byid))