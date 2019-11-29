from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pymongo
import scipy
import numpy as np
from db_interface import DbInterface,DbRecord
import time
from joblib import dump, load

saved_models = 'svm_models_trained'

def getDbName(datasetName, modelName):
    return datasetName + '_' + modelName

def createSvmModel(kernelType):
    return svm.SVC(kernel=kernelType, gamma='scale')

def transformFeature(features, trType=None):
    tr = lambda x: x.flatten()
    if trType == 'pca':
        features = [x.flatten() for x in features]
        pca = IncrementalPCA(n_components=100, batch_size = 200).fit(features)
        return pca.transform(features), pca
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
        if features[0].size > (2048 * 2048): # Reduce using PCA since the matrix is too large
            print('Size of matrix is {}, applying PCA to reduce dimensions'.format(features[0].size)) 
            features = PCA(n_components=0.95).fit_transform([x.flatten() for x in features])
        tr = matrixLog
            
    return [tr(feature) for feature in features]

def writeResultToFile(outfile, ds, model, layer, kernel, transform, acc):
    with open(outfile, 'a') as f:
        f.write('{},{},{},{},{},{}\n'.format(ds, model, layer, kernel, transform, acc))
        f.close()

def shuffle(x,y):
    c = list(zip(x,y))
    np.random.shuffle(c)
    return zip(*c)

def train_val_split_closedset(x, y, percent=.25, min_entries=3, seed=12924):
    '''
        x - list of features
        y - list of ids corresponding to each feature 
        percent - percentage of each id to split
        min_entries - filter out any ids which have less than this number of images
        returns - shuffled ids for the features
                  train_x_indices,train_y,test_x_indices,test_y
                  *_y contain ids mapped to the features
                  *_x contain the indices of the features
    '''
    # For Id:Images map
    assert len(x) == len(y)
    assert len(x) > 0
    id_indices_map = {}
    for i in range(len(x)):
        if not y[i] in id_indices_map:
            id_indices_map[y[i]] = []
        id_indices_map[y[i]].append(i)
    
    # Filter out ids that don't have min number of images
    id_indices_map = {k:v for k,v in id_indices_map.items() if len(v) >= min_entries}
    
    # For each id, split percent of entries randomly and separate them
    train_x_indices = []
    test_x_indices = []
    train_y = []
    test_y = []

    np.random.seed(seed)
    for i in id_indices_map.keys():
        n = max( 1, int( np.ceil(percent * len(id_indices_map[i])) ) )

        l = len(id_indices_map[i])
        pick = np.random.choice(range(l), n) # pick n elements at random
        te_x = [id_indices_map[i][x] for x in pick]
        tr_x = [id_indices_map[i][x] for x in range(l) if x not in pick] # Remaining goes to training set
        te_y = [i for _ in range(len(te_x))] # All Test ids = i
        tr_y = [i for _ in range(len(tr_x))] # All Train ids = i
        
        train_x_indices.extend(tr_x)
        test_x_indices.extend(te_x)
        train_y.extend(tr_y)
        test_y.extend(te_y)

    train_x_indices,train_y = shuffle(train_x_indices,train_y)
    test_x_indices,test_y = shuffle(test_x_indices,test_y)
    
    return train_x_indices,train_y,test_x_indices,test_y

def train_svm_for_layer(client, datasetName, modelName, layer, outfile):
    dbCursor = client.getDB(getDbName(datasetName, modelName))

    # print('Training Size: {}, Test Size: {}'.format(len(X_train), len(X_test)))
    max_acc = 0
    best_tr_type = None
    best_kernel_type = None
    sum_times = 0
    count = 0
    for trType in ['pca']: # trType in [None, 'pca', 'logm']:
        # The 4 statements can be moved outside the loop, but then they would be continuously in memory
        # Add them here so that they can be safely deleted to make memory for expensive SVM operations
        records = [dbCursor.getRecord(layer, recordId) for recordId in dbCursor.getRecordIds(layer)]
        obs = np.array([x.feature for x in records])
        target = [x.animalId for x in records]
        del records

        try:
            samples,pca_model = transformFeature(obs, trType)
        except ValueError as e:
            print('Transformation {} couldnt be applied for layer {}: {}, skipping it'.format(trType, layer, e))
            writeResultToFile(outfile, datasetName, modelName, layer, '-', trType, -1.0)
            continue

        del obs

        # X_train, X_test, y_train, y_test = train_test_split(samples, target, test_size=0.25)
        X_train_indices, y_train, X_test_indices, y_test = train_val_split_closedset(samples, target)

        for kernelType in ['linear', 'rbf']:
            svmModel = createSvmModel(kernelType)
            try:
                svmModel.fit([samples[x] for x in X_train_indices] , y_train)
            except ValueError as e:
                print('Failed to fit SVM model with kernel {}, for layer {}, transform {}: {}, skipping'.format(kernelType, layer, trType, e))
                writeResultToFile(outfile, datasetName, modelName, layer, kernelType, trType, -1.0)
                continue

            start = time.time()
            y_pred = svmModel.predict([samples[x] for x in X_test_indices])
            sum_times += (time.time() - start)
            count += len(X_test_indices)

            acc = metrics.accuracy_score(y_test, y_pred)
            # print('Layer Name: {}, Kernel Type: {}, Transofrm: {}, Accuracy: {}'
            #          .format(layer, kernelType, trType, acc))
            writeResultToFile(outfile, datasetName, modelName, layer, kernelType, trType, acc)
            if acc > max_acc:
                # print('New max accuracy using layer {}, Kernel: {}, Transform {}. Value: {}'.format(layer, kernelType, trType, acc))
                best_tr_type = trType
                best_kernel_type = kernelType
                max_acc = acc
                # Save model to disk
                lr_name = layer.replace('/', '&')
                dump((svmModel, pca_model, set(target)), '{}/{}-{}-{}-{}-{}.model'.format(saved_models, datasetName, modelName, lr_name, kernelType, trType))

    avg_time = sum_times
    if count > 0:
        avg_time = sum_times / count
    return max_acc, best_tr_type, best_kernel_type, avg_time

def find_best_svm_model(dsName, modelNames):
    # modelNames = ['alexnet', 'googlenet']
    # modelNames = ['alexnet']
    # datasetNames = ['amur']
    client = DbInterface()

    overall_max_acc = -1
    max_acc_layer = None
    max_acc_transform = None
    max_acc_model = None
    best_kernel_type = None

    outfile = 'svm_accuracy.csv'

    with open(outfile, 'w') as f:
        f.write('Dataset Name,Model Name,Layer Name,Kernel Type,Transform Type,Accuracy\n')
        f.close()

    sum_times = 0
    count = 0
    for modelName in modelNames:
        dbCursor = client.getDB(getDbName(dsName, modelName))
        layerNames = dbCursor.getCollectionNames()
        # layerNames = ['inception_4d/relu_1x1']
        for layer in layerNames:
            print('Starting Training for model {}, over layer {}'.format(modelName, layer))
            acc, trType, kernelType,times = train_svm_for_layer(client, dsName, modelName, layer, outfile)
            count += 1
            sum_times += times
            if acc > overall_max_acc:
                # print('Global Max accuracy being set, Value: {}, Model: {}, Layer: {}, Transform: {}'.
                #          format(acc, modelName, layer, trType))
                overall_max_acc = acc
                max_acc_layer = layer
                max_acc_model = modelName
                max_acc_transform = trType
                best_kernel_type = kernelType

    avg_time = sum_times
    if count > 0:
        avg_time = sum_times/count
    return overall_max_acc,max_acc_layer,max_acc_model,max_acc_transform,best_kernel_type,avg_time

if __name__ == '__main__':
    modelNames = ['alexnet', 'googlenet', 'resnet50']
    datasets = ['jaguar']

    for ds in datasets:
        acc,layer,model,transform,kernel,avg_time = find_best_svm_model(ds, modelNames)
        print('\n\nModel testing concluded for Dataset: {}.\n\n Max Accuracy: {:.3f}, Layer: {}, Model: {}, Transform: {}, Kernel: {}'
                .format(ds,acc,layer,model,transform,kernel) )

        print('Average Single image SVM prediction time: {:.4f}s'.format((avg_time)))#+avg_time2)/2))

        # Remove all other models for DS
        lr_name = layer.replace('/', '&')
        best_model = '{}/{}-{}-{}-{}-{}.model'.format(saved_models, ds, model, lr_name, kernel, transform)
        for f in os.listdir(saved_models):
            if not f == best_model:
                os.remove(f)
