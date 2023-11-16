from myfunc import *
from numpy import *
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--trainPath', type=str, default='/tmp2/b10902005/fintech/final/dataset_1st/new_train.csv')
parser.add_argument('--outDir', type=str, default='1113')
args = parser.parse_args()

def getRes(y0, y1):
	res=[[0, 0], [0, 0]]
	y0=[0 if i==1 else 1 for i in y0]
	y1=[0 if i==1 else 1 for i in y1]
	for i in range(len(y0)):
		res[y0[i]][y1[i]]+=1
	return res

#hotEncoder=['contp', 'etymd', 'insfg', 'bnsfg', 'stocn', 'scity', 'hcefg', 'csmcu']
#lbEncoder=['txkey', 'chid', 'cano', 'mchno', 'acqic', 'mcc', 'ecfg', 'stscd', 'ovrlt', 'flbmk', 'flg_3dsmk']

data=ReadCSV(args.trainPath)
#numeric_features = ['locdt', 'loctm', 'conam', 'iterm', 'flam1', 'csmam']
#category_features = ['chid', 'cano', 'contp', 'etymd', 'mchno', 'acqic', 'mcc', 'ecfg', 'insfg', 'bnsfg', 'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'flg_3dsmk']
#num_numeric_features = len(numeric_features)
#output_size = 1
#output_dir = args.output_dir
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)
#vocab_file = f'{output_dir}/vocab.json'
#param_file = f'{output_dir}/param.json'
#aveParam(args, param_file)
#he, le={i:[] for i in hotEncoder}, {i:[] for i in lbEncoder}
#for i in data:
#	for j in hotEncoder:
#		he[j].append(i[j])
#	for j in lbEncoder:
#		le[j].append(i[j])
#he={i:LabelEncoder().fit_transform(he[i]) for i in he}
#le={i:LabelEncoder().fit_transform(le[i]) for i in le}
#for i in range(len(data)):
#	for j in hotEncoder:
#		data[i][j]=he[j]
#	for j in lbEncoder:
#		data[i][j]=le[j]

#data=data[:1000]
#X=X[:1000]
#y=y[:1000]
#print('cut done')
X=[[i[j] for j in i if j!='label' and j!='txkey'] for i in data]
print('remove label and txkey done')
#print([i['label'] for i in data])
y=[1 if i['label']=='0' else -1 for i in data]
print('y done')
model=IsolationForest(random_state=49).fit(X)
#print(model.predict(X))
#print(y)
res=getRes(y, model.predict(X))
print(res)
print(2*res[1][1]/(2*res[1][1]+res[1][0]+res[0][1]))
