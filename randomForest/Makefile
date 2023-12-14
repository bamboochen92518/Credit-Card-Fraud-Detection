path=/tmp2/b10902085/fintech/input
ver=_logi.csv
all:
	cp ../myfunc.py ./
	python3 genSmall.py --cnt=100000 --inPath=$(path)/new_train$(ver) --outPath=small_new_train.csv
	python3 genSmall.py --cnt=100000 --inPath=$(path)/new_valid$(ver) --outPath=small_new_valid.csv
	python3 genSmall.py --cnt=100000 --inPath=$(path)/new_test$(ver) --outPath=small_new_test.csv
	python3 train.py --trainPath=small_new_train.csv --validPath=small_new_valid.csv --testPath=small_new_test.csv
#small:
#	cp ../myfunc.py ./
#	python3 genSmall.py --cnt=100000 --inPath=/tmp2/b10902005/fintech/final/dataset_2nd/new_train.csv --outPath=small_new_train.csv
#	python3 train.py --trainPath=small_new_train.csv --testPath=/tmp2/b10902005/fintech/final/dataset_2nd/new_test.csv
big:
	cp ../myfunc.py ./
	python3 train.py --trainPath=$(path)/new_train$(ver) --validPath=$(path)/new_valid$(ver) --testPath=$(path)/new_test$(ver)
upload:
	cp ../myfunc.py ./
	python3 train.py --trainPath=$(path)/new_train$(ver) --testPath=/tmp2/b10902005/fintech/final/dataset_2nd/new_test.csv --validToTrain=1
clean:
	rm small_new_train.csv
	rm small_new_valid.csv
	rm small_new_test.csv
