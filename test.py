import csv
from myfunc import ReadCSV, WriteCSV

csv_in_path = '../final/dataset_1st/public_processed.csv'
csv_out_path = 'prediction.csv'

data = ReadCSV(csv_in_path)

write_data = list()

for row in data:
    tmp = [row["txkey"], 1]
    write_data.append(tmp)

WriteCSV(csv_out_path, write_data)
