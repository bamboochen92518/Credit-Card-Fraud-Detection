all:
	cp ../myfunc.py ./
	python3 genSmall.py --cnt=100000 --inPath=/tmp2/b10902005/fintech/final/dataset_2nd/new_train.csv --outPath=small_new_train.csv
	python3 genSmall.py --cnt=100000 --inPath=/tmp2/b10902005/fintech/final/dataset_2nd/new_test.csv --outPath=small_new_test.csv
	python3 train.py --trainPath=small_new_train.csv --testPath=small_new_test.csv
small:
	cp ../myfunc.py ./
	python3 genSmall.py --cnt=1000000 --inPath=/tmp2/b10902005/fintech/final/dataset_2nd/new_train.csv --outPath=small_new_train.csv
	python3 train.py --trainPath=small_new_train.csv --testPath=/tmp2/b10902005/fintech/final/dataset_2nd/new_test.csv
big:
	cp ../myfunc.py ./
	python3 train.py --trainPath=/tmp2/b10902005/fintech/final/dataset_2nd/new_train.csv --testPath=/tmp2/b10902005/fintech/final/dataset_2nd/new_test.csv
clean:
	rm small_new_train.csv
	rm small_new_valid.csv
	rm small_new_test.csv
