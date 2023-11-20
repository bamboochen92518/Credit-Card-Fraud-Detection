all:
	cp ../myfunc.py ./
	python3 genSmall.py --cnt=100000 --inPath=/tmp2/b10902005/fintech/final/dataset_1st/new_train_ver2.csv --outPath=small_new_train.csv
	python3 genSmall.py --cnt=100000 --inPath=/tmp2/b10902005/fintech/final/dataset_1st/new_test_ver2.csv --outPath=small_new_test.csv
	python3 train.py --trainPath=small_new_train.csv --testPath=small_new_test.csv
small:
	cp ../myfunc.py ./
	python3 genSmall.py --cnt=1000000 --inPath=/tmp2/b10902005/fintech/final/dataset_1st/new_train_ver2.csv --outPath=small_new_train.csv
	python3 train.py --trainPath=small_new_train.csv --testPath=/tmp2/b10902005/fintech/final/dataset_1st/new_test_ver2.csv
big:
	cp ../myfunc.py ./
	python3 train.py --trainPath=/tmp2/b10902005/fintech/final/dataset_1st/new_train_ver2.csv --testPath=/tmp2/b10902005/fintech/final/dataset_1st/new_test_ver2.csv
clean:
	rm small_new_train.csv
	rm small_new_test.csv
