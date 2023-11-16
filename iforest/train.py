from myfunc import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--trainPath', type=str, default='../../final/dataset_1st/training.csv')
parser.add_argument('--outDir', type=str, default='1113')
args = parser.parse_args()


data=ReadCSV(args.trainPath)
