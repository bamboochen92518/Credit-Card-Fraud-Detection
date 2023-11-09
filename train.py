import csv
from tqdm import tqdm
from myfunc import ReadCSV, ObserveKey, Sort_Data_with_CANO_and_Time
from prettytable import PrettyTable
csv_file_path = 'dataset_1st/training.csv'

data = ReadCSV(csv_file_path)

data = Sort_Data_with_CANO_and_Time(data)
