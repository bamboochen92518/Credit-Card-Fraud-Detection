from myfunc import *
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--csv0', type=str)
parser.add_argument('--csv1', type=str)
args = parser.parse_args()

data0=ReadCSV(args.csv0)
data1=ReadCSV(args.csv1)
assert(len(data0)==len(data1))
res=[[0, 0], [0, 0]]
for i in range(len(data0)):
	res[int(data0[i]['pred'])][int(data1[i]['pred'])]+=1
print(res)
print('csv0 contamination rate =', sum(res[1])/(sum(res[0])+sum(res[1])))
print('csv1 contamination rate =', (res[0][1]+res[1][1])/(sum(res[0])+sum(res[1])))

