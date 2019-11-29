from db_interface import DbInterface, DbRecord

db = DbInterface().getDB('amur_alexnet')
recs = []
target = []
for recId in db.getRecordIds('conv1'):
    rec = db.getRecord('conv1', recId)
    recs.append(rec.feature)
    target.append(rec.animalId)

def convertToFeature(f):
    i = 1
    res = ''
    for r in f.flatten():
        res += '{}:{} '.format(i, r)
        i += 1
    return res.strip()

with open('train_1.txt', 'w') as f:
    for i in range(5):
        f.write('{} {}\n'.format(target[i], convertToFeature(recs[i])))
    f.close()

with open('test_1.txt', 'w') as f:
    for i in range(5,10):
        f.write('{} {}\n'.format(target[i], convertToFeature(recs[i])))
    f.close()

