from extract_features_and_store_to_mongo import extractFeaturesForImages
from train_validate_svm import find_best_svm_model
import sys

args = sys.argv
if len(args) < 2:
    print('At least 1 image folder required')
    exit(1)

extractFeaturesForImages(args[1:])
print()
print('All features extracted and stored to DB, beginning SVM training')

dsNames = set()
for folder in args[1:]:
    if 'amur' in folder:
        dsNames.add('amur')
    elif 'elp' in folder:
        dsNames.add('elp')

modelNames = ['alexnet']
for dsName in dsNames:
    acc,layer,model,transform,kernel = find_best_svm_model(dsName, modelNames)
    print('\n\nModel testing concluded for Dataset {}.\n\n Max Accuracy: {}, Layer: {}, Model: {}, Transform: {}, Kernel: {}'
        .format(dsName,acc,layer,model,transform,kernel) )
