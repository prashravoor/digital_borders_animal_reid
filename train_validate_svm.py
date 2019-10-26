from sklearn import svm
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pymongo
import scipy
import numpy as np
from db_interface import DbInterface,DbRecord

def getDbName(datasetName, modelName):
    return datasetName + '_' + modelName

def createSvmModel(kernelType):
    return svm.SVC(kernel=kernelType, gamma='scale')

def transform(features, trType=None):
    tr = lambda x: x.flatten()
    if trType == 'pca':
        features = [x.flatten() for x in features]
        tr = lambda features: PCA(n_components=0.95).fit_transform(features)
        return tr(features)
    elif trType == 'logm':
        def matrixLog(f):
            if not len(f.shape) == 2:
                # Reshape into a 2d matrix
                lastDim = f.shape[-1]
                f = f.reshape((int(f.size/lastDim), lastDim))
            # Normalize f
            abs_sum = abs(np.sum(f))
            f /= (max(10e-8, abs_sum))

            # Converting f to a symmetric matrix (square)
            f = f @ f.T
            f = f + 10e-5*np.identity(f.shape[0]) # Add a small Identity matrix to prevent f from being all 0s
            res = scipy.linalg.logm(f) # Matrix logarithm
            res = np.tril(res) # Retain only lower triangle elements, since this is a symmetric matrix
            return res.flatten()
        tr = matrixLog
            
    return [tr(feature) for feature in features]

def writeResultToFile(outfile, ds, model, layer, kernel, transform, acc):
    with open(outfile, 'a') as f:
        f.write('{},{},{},{},{},{}\n'.format(ds, model, layer, kernel, transform, acc))
        f.close()

def train_svm_for_layer(client, datasetName, modelName, layer, outfile):
    dbCursor = client.getDB(getDbName(datasetName, modelName))


    # print('Training Size: {}, Test Size: {}'.format(len(X_train), len(X_test)))
    max_acc = 0
    best_tr_type = None
    best_kernel_type = None
    for trType in [None, 'pca', 'logm']:
        # The 4 statements can be moved outside the loop, but then they would be continuously in memory
        # Add them here so that they can be safely deleted to make memory for expensive SVM operations
        records = [dbCursor.getRecord(layer, recordId) for recordId in dbCursor.getRecordIds(layer)]
        obs = np.array([x.feature for x in records])
        target = [x.animalId for x in records]
        del records

        try:
            samples = transform(obs, trType)
        except ValueError as e:
            print('Transformation {} couldnt be applied for layer {}: {}, skipping it'.format(trType, layer, e))
            writeResultToFile(outfile, datasetName, modelName, layer, '-', trType, -1.0)
            continue

        del obs

        X_train, X_test, y_train, y_test = train_test_split(samples, target, test_size=0.25)
        del samples

        for kernelType in ['linear', 'rbf']:
            svmModel = createSvmModel(kernelType)
            try:
                svmModel.fit(X_train, y_train)
            except ValueError as e:
                print('Failed to fit SVM model with kernel {}, for layer {}, transform {}: {}, skipping'.format(kernelType, layer, trType, e))
                writeResultToFile(outfile, datasetName, modelName, layer, kernelType, trType, -1.0)
                continue

            y_pred = svmModel.predict(X_test)
            acc = metrics.accuracy_score(y_test, y_pred)
            print('Layer Name: {}, Kernel Type: {}, Transofrm: {}, Accuracy: {}'
                      .format(layer, kernelType, trType, acc))
            writeResultToFile(outfile, datasetName, modelName, layer, kernelType, trType, acc)
            if acc > max_acc:
                print('New max accuracy using layer {}, Kernel: {}, Transform {}. Value: {}'.format(layer, kernelType, trType, acc))
                best_tr_type = trType
                best_kernel_type = kernelType

    return max_acc, best_tr_type, best_kernel_type

def find_best_svm_model():
    modelNames = ['alexnet', 'googlenet']
    datasetNames = ['amur']
    client = DbInterface()

    overall_max_acc = 0
    max_acc_layer = None
    max_acc_transform = None
    max_acc_model = None
    best_kernel_type = None

    outfile = 'svm_accuracy.csv'

    with open(outfile, 'w') as f:
        f.write('Dataset Name,Model Name,Layer Name,Kernel Type,Transform Type,Accuracy\n')
        f.close()

    for dsName,modelName in [(x,y) for x in datasetNames for y in modelNames]:
        dbCursor = client.getDB(getDbName(dsName, modelName))
        layerNames = dbCursor.getCollectionNames()
        for layer in layerNames:
            print('Starting Training for model {}, over layer {}'.format(modelName, layer))
            acc, trType, kernelType = train_svm_for_layer(client, dsName, modelName, layer, outfile)
            if acc > overall_max_acc:
                print('Global Max accuracy being set, Value: {}, Model: {}, Layer: {}, Transform: {}'.
                          format(acc, modelName, layer, trType))
                overall_max_acc = acc
                max_acc_layer = layer
                max_acc_model = modelName
                max_acc_transform = trType
                best_kernel_type = kernelType

    return overall_max_acc,max_acc_layer,max_acc_model,max_acc_transform,best_kernel_type

if __name__ == '__main__':
    acc,layer,model,transform,kernel = find_best_svm_model()
    print('\n\nModel testing concluded.\n\n Max Accuracy: {}, Layer: {}, Model: {}, Transform: {}, Kernel: {}'
            .format(acc,layer,model,transform,kernel) )
