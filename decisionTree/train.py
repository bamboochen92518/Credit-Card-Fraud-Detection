from myfunc import *
from numpy import *
from time import *
#from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--trainPath', type=str, default='/tmp2/b10902005/fintech/final/dataset_2nd/new_train.csv')
parser.add_argument('--validPath', type=str, default='/tmp2/b10902005/fintech/final/dataset_2nd/new_valid.csv')
parser.add_argument('--testPath', type=str, default='/tmp2/b10902005/fintech/final/dataset_2nd/new_test.csv')
parser.add_argument('--outDir', type=str, default='/tmp2/b10902085/fintech/')
parser.add_argument('--validToTrain', type=int, default=0)
parser.add_argument('--removeSome', type=int, default=1)
args = parser.parse_args()
outFile=args.outDir+strftime('%b%d-%H%M%S')

def getRes(y0, y1):
	res=[[0, 0], [0, 0]]
	y0=[0 if i==1 else 1 for i in y0]
	y1=[0 if i==1 else 1 for i in y1]
	for i in range(len(y0)):
		res[y0[i]][y1[i]]+=1
	return res

torm=['label', 'txkey'] if args.removeSome==0 else ['label', 'txkey', 'chid', 'mchno', 'iterm', 'bnsfg', 'scity', 'csmcu', 'csmam'] # label to be removed

data=ReadCSV(args.trainPath)
print([i for i in data[0]])
#X=[[i[j] for j in i if j not in torm] for i in data]
X=[[i[j] for j in i if j not in torm]+[float(i['cano'])>0] for i in data]
print('remove label and txkey done')
y=[1 if i['label']=='0' else -1 for i in data]
print('y done')
data=ReadCSV(args.validPath)
#Xv=[[i[j] for j in i if j not in torm and j!='txkey'] for i in data]
Xv=[[i[j] for j in i if j not in torm]+[float(i['cano'])>0] for i in data]
print('remove label and txkey done')
yv=[1 if i['label']=='0' else -1 for i in data]
print('yv done')
data=ReadCSV(args.testPath)
#Xt=[[i[j] for j in i if j not in torm] for i in data]
Xt=[[i[j] for j in i if j not in torm]+[float(i['cano'])>0] for i in data]
print('remove label and txkey done')

if args.validToTrain:
	X+=Xv
	y+=yv

#model=RandomForestClassifier(n_estimators=100, criterion='gini', verbose=True)
model=DecisionTreeClassifier()
model.fit(X, y)
y1=model.predict(X)
res=getRes(y, y1)
print(res)
yv1=model.predict(Xv)
res=getRes(yv, yv1)
print(res)
ev=2*res[1][1]/(2*res[1][1]+res[1][0]+res[0][1])
print('eval =', ev)
outFile+='_eval_'+str(ev)
yt=model.predict(Xt)
crate=(len(yt)-sum(yt))/(len(yt)*2)
outFile+='_crate_'+str(crate)+'.csv'
WriteCSV(outFile, [[data[i]['txkey'], 0 if yt[i]==1 else 1] for i in range(len(yt))])
print('contamination rate =', crate)
