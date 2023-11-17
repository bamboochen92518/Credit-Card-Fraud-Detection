from myfunc import *
from numpy import *
from time import *
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--trainPath', type=str, default='/tmp2/b10902005/fintech/final/dataset_1st/new_train.csv')
parser.add_argument('--testPath', type=str, default='/tmp2/b10902005/fintech/final/dataset_1st/new_test.csv')
parser.add_argument('--outDir', type=str, default='/tmp2/b10902085/fintech/')
args = parser.parse_args()
outFile=args.outDir+strftime('%b%d-%H%M%S')+'_ein_'

def getRes(y0, y1):
	res=[[0, 0], [0, 0]]
	y0=[0 if i==1 else 1 for i in y0]
	y1=[0 if i==1 else 1 for i in y1]
	for i in range(len(y0)):
		res[y0[i]][y1[i]]+=1
	return res

data=ReadCSV(args.trainPath)
X=[[i[j] for j in i if j!='label' and j!='txkey'] for i in data]
print('remove label and txkey done')
y=[1 if i['label']=='0' else -1 for i in data]
print('y done')
data=ReadCSV(args.testPath)
Xt=[[i[j] for j in i if j!='txkey'] for i in data]
print('remove label and txkey done')

contRate=(len(y)-sum(y))/(len(y)*2)
clf=IsolationForest(random_state=49, verbose=True, contamination=contRate)
model=clf.fit(X+Xt)
res=getRes(y, model.predict(X))
print(res)
ein=2*res[1][1]/(2*res[1][1]+res[1][0]+res[0][1])
print(ein)
outFile+=str(ein)+'.csv'
yt=model.predict(Xt)
WriteCSV(outFile, [[data[i]['txkey'], 0 if yt[i]==1 else 1] for i in range(len(yt))])
print('contamination rate =', (len(yt)-sum(yt))/(len(yt)*2))
