import pymongo as pm
import pickle
from bson.binary import Binary
from bson.objectid import ObjectId

class DbRecord:
    def __init__(self, animalId, imageFile, feature):
        self.animalId = animalId
        self.imageFile = imageFile
        self.feature = feature

    def convertToDoc(self):
        self.feature = Binary( pickle.dumps(self.feature, protocol=2) )

    def convertFromDoc(self):
        self.feature = pickle.loads(self.feature)
        

class DbInterface:
    def __init__(self, host='localhost', port=27017):
        self.client = pm.MongoClient(host, port)

    def getDB(self, dbName):
        return DbCursor(self.client, dbName)


class DbCursor:
    def __init__(self, client, dbName):
        self.client = client
        self.dbName = dbName
        self.dbCursor = client[dbName]
        
    #def getCollection(self, collectionName):
    #    return dbCursor[collectionName]

    def getCollectionNames(self):
        return self.dbCursor.list_collection_names()

    def dropCollection(self, collectionName):
        self.dbCursor[collectionName].drop()

    def insertAsync(self, collectionName, record):
        collection = self.dbCursor[collectionName]
        record.convertToDoc()
        collection.insert_one(vars(record))

    def getRecordIds(self, collectionName):
        val = self.dbCursor[collectionName].find({}, {"_id" : 1})
        for v in val:
            yield v['_id']

    def getRecord(self, collectionName, recordId):
        rec = self.dbCursor[collectionName].find_one({"_id" : ObjectId(recordId)})
        if not rec:
            return None
        res = DbRecord(rec['animalId'], rec['imageFile'], rec['feature'])
        res.convertFromDoc()
        return res
