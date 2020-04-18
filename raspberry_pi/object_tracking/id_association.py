from im_utils import iou, extractImageFromBoundingBox
import numpy as np
from collections import namedtuple, defaultdict
from sklearn.metrics.pairwise import euclidean_distances
import time

DeviceDetection = namedtuple('DeviceDetection', 'detections timestamp')
#LABELS = {0: 'Human', 1: 'Tiger', 2: 'Elephant', 3: 'Jaguar', 4: 'Human'}
LABELS = {0: 'Human', 1: 'Human', 2: 'Elephant', 3: 'Jaguar', 4: 'Human'}

class FeatureVectorDatabase:
    def __init__(self):
        self.perClassFeatureDb = defaultdict(list)

    def getCurrentKnownIds(self, classid):
        ids = set()
        for entry in self.perClassFeatureDb[classid]:
            ids.add(entry[1]) # Tuples are like (feature_vector, id)

        return ids

    def addFeatureVector(self, classid, vector, id):
        self.perClassFeatureDb[classid].append((vector, id))
        
    def getKnownFeatureVectors(self, classid):
        return self.perClassFeatureDb[classid]        

class IdAssociator:
    def __init__(self, featureExtractorMap, featureDb):
        self.featureExtractorMap = featureExtractorMap
        self.featureDb = featureDb
        self.perDeviceDetHistory = dict()
        #self.SIMILARITY_THRESH = 45.0
        self.SIMILARITY_THRESH = 750.0
        self.devDetMap = dict()


    def generateNewId(self, classid):
        classid = int(classid)
        knownDets = self.featureDb.getCurrentKnownIds(classid) 
        return len(knownDets) # Ids are contiguous, so return length of set, which would be the new id

    def getIdsForFeatureVectors(self, classid, vectors):
        '''
        returns identities for each vector specified
        '''
        knownFeatures = self.featureDb.getKnownFeatureVectors(classid)

        if len(knownFeatures) == 0:
            return [x for x in range(len(vectors))]

        knownIds = set([x[1] for x in knownFeatures])
        knownVectors = np.array([x[0] for x in knownFeatures])
        distMat = euclidean_distances(vectors, knownVectors)
        identities = []
        countmap = defaultdict(int)
        for f in knownFeatures:
            countmap[f[1]] += 1
        minNumSamples = min([v for _,v in countmap.items()])

        for i in range(len(vectors)):
            vec = distMat[i] 
            print(np.min(vec), np.max(vec))
            print(minNumSamples)
            if np.min(vec) > self.SIMILARITY_THRESH and minNumSamples > 3:
                # Return new id
                identities.append(len(knownIds))
            else:
                '''
                index = np.argmin(vec) # Get index of min distance vector
                identities.append(knownFeatures[index][1]) # Corresponding identity is assigned
                '''
                # Consider ids in decreasing order of likelihood
                indices = np.argsort(vec)
                added = False
                for i in indices:
                    newid = knownFeatures[i][1]
                    if newid in identities: # Already assigned
                        continue
                    identities.append(newid) # Corresponding identity is assigned
                    added = True
                    break

                if not added:
                    identities.append(self.generateNewId(classid)) # New identity is assigned

        assert len(identities) == len(vectors), 'Identites: {} != Vectors: {}'.format(identities, len(vectors))
        return identities

    def update(self, devName, image, detections):
        '''
            if first time detection, return new ids for all
            or, check feature vector db, and return known ids
            if prev frame exists for device, map iou of each pair of boxes
            if there are n-1 iou overlaps, then return identities as is
            if more than one than one pair of objects are overlapping, match identities based on feature vectors
            in all cases, update feature vectorDb

            Cost ineffective way
            for each detection, simply extract feature vector, match closest, and return id
        '''

        prev = []
        if devName in self.devDetMap:
            prev = self.devDetMap[devName] # List of tuples like [(idf, detection)]

        detIndexMap = dict() # Contains map of detectionIndex : identitiy
        not_matched = []
        for i in range(len(detections)):
            det = detections[i]
            ious = np.array([iou(det.bounding_box, x[1].bounding_box) for x in prev if x[1].classid == det.classid])
            non_zero = np.count_nonzero(ious > 0.1) # Greater than 10% overlap
            if non_zero == 1: # Matching box found
                # result.append((prev[np.argmax(ious)][0], det)) # argmax has index of max IOU overlap
                detIndexMap[i] = prev[np.argmax(ious)][0]
                print('IOU based match success!')
            else:
                not_matched.append(i)
                # It is much easier to simply extract feature vector and get identity here
                # due to some tensorflow bug, feature extraction takes 20s+ when called in for loop
                # so handle these detections later, with different kind of for loop and lot more complexity, but faster

        if len(not_matched) > 0:
            imageMap = defaultdict(list) # Images grouped by their class ids
            print('Starting Feature Extraction for total {} images'.format(len(not_matched)))
            start = time.time()
            for i in not_matched:
                det = detections[i]
                imageMap[(i, int(det.classid))].append(extractImageFromBoundingBox(image, det.bounding_box))

            # Run feature extraction one class group at a time
            for k,v in imageMap.items():
                vectors = np.array(self.featureExtractorMap[k[1]].extractMultiple(v)) # k[1] is class id
                identities = self.getIdsForFeatureVectors(k[1], vectors)
                for i in range(len(identities)):
                    self.featureDb.addFeatureVector(k[1], vectors[i], identities[i]) # Update database with new features
                    detIndexMap[k[0]] = identities[i] # k[0] is the detection index

            print('Feature Extraction time: {:.4f}s'.format(time.time() - start))

        self.devDetMap[devName] = [(v, detections[k]) for k,v in detIndexMap.items()] # detIndexMap -> detetionIndex : identity

        return [v for _,v in detIndexMap.items()]

        '''
        print('Found matching IOUs for {} detections, {} not found'.format(len(cur), len(not_matched)))

        for det in not_matched:
            imageMap[int(det.classid)].append(extractImageFromBoundingBox(image, det.bounding_box))

        for k,v in imageMap.items():
            vectors = np.array(self.featureExtractorMap[k].extractMultiple(v))
            identities = self.getIdsForFeatureVectors(k, vectors)
            for i in range(len(identities)):
                self.featureDb.addFeatureVector(k, vectors[i], identities[i]) # Update database with new features
                idMap[k].append(identities[i])

        # Return ids in same order as detections
        for det in detections:
            result.append(idMap[det.classid][0])
            del idMap[det.classid][0]

        return result
        '''
