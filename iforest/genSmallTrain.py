from myfunc import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--inPath', type=str, default='/tmp2/b10902005/fintech/final/dataset_1st/new_train.csv')
parser.add_argument('--outPath', type=str, default='small_new_train.csv')
parser.add_argument('--cnt', type=int, default=1000)
args = parser.parse_args()

data=[]
for line in open(args.inPath, 'r'):
	data.append(line)
	if len(data)>=args.cnt:
		break

fp=open(args.outPath, 'w')
for i in data:
	fp.write(i)
