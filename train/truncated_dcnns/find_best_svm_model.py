from extract_features_and_store_to_mongo import extractFeaturesForImages
from train_validate_svm import find_best_svm_model
import sys
import os

args = sys.argv
if len(args) < 3:
    print('Usage: cmd <Object Detector> <Folder name 1> <Folder name 2> ... [max images]')
    exit(1)

saved_models = 'svm_models_trained'
#modelNames = ['googlenet', 'resnet50', 'alexnet']
modelNames = ['alexnet', 'googlenet']
extractFeaturesForImages(modelNames,args[1:])
print()
print('All features extracted and stored to DB, beginning SVM training')

dsNames = set()
for folder in args[1:]:
    if 'amur' in folder:
        dsNames.add('amur')
    elif 'elp' in folder:
        dsNames.add('elp')
    else:
        dsNames.add('jaguar')

for dsName in dsNames:
    acc,layer,model,transform,kernel,avg_time = find_best_svm_model(dsName, modelNames)
    print('\n\nModel testing concluded for Dataset {}.\n\n Max Accuracy: {}, Layer: {}, Model: {}, Transform: {}, Kernel: {}'
        .format(dsName,acc,layer,model,transform,kernel) )
    print('Average prediction time for single image: {:.4f}s'.format(avg_time))

    # Remove all other models for DS
    #lr_name = layer.replace('/', '&')
    #best_model = os.path.join(saved_models,'{}-{}-{}-{}-{}.model'.format(dsName, model, lr_name, kernel, transform))
    #for f in os.listdir(saved_models):
    #    if not f == best_model:
    #       os.remove(os.path.join(saved_models,f))
