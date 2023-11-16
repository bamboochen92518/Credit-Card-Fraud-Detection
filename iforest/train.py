from myfunc import *
from numpy import *
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--trainPath', type=str, default='/tmp2/b10902005/fintech/final/dataset_1st/small_train.csv')
parser.add_argument('--outDir', type=str, default='1113')
args = parser.parse_args()

#hotEncoder=['contp', 'etymd', 'insfg', 'bnsfg', 'stocn', 'scity', 'hcefg', 'csmcu']
#lbEncoder=['txkey', 'chid', 'cano', 'mchno', 'acqic', 'mcc', 'ecfg', 'stscd', 'ovrlt', 'flbmk', 'flg_3dsmk']

data=ReadCSV(args.trainPath)
numeric_features = ['locdt', 'loctm', 'conam', 'iterm', 'flam1', 'csmam']
category_features = ['chid', 'cano', 'contp', 'etymd', 'mchno', 'acqic', 'mcc', 'ecfg', 'insfg', 'bnsfg', 'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'flg_3dsmk']
num_numeric_features = len(numeric_features)
output_size = 1
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
vocab_file = f'{output_dir}/vocab.json'
param_file = f'{output_dir}/param.json'
SaveParam(args, param_file)
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

X=[[i[j] for j in i if j!='label'] for i in data]
print(len(X))
y=[1 if i['label']==0 else -1 for i in data]
model=IsolationForest(random_state=0).fit(X)
print(model.predict(X))
print(y)
