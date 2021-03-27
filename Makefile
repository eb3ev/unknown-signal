all: basic adv noise

basic: basic1 basic2 basic3 basic4

basic1:
	python lsr.py train_data/basic_1.csv

basic2:
	python lsr.py train_data/basic_2.csv

basic3:
	python lsr.py train_data/basic_3.csv

basic4:
	python lsr.py train_data/basic_4.csv

adv: adv1 adv2 adv3

adv1:
	python lsr.py train_data/adv_1.csv

adv2:
	python lsr.py train_data/adv_2.csv

adv3:
	python lsr.py train_data/adv_3.csv

noise: noise1 noise2 noise3

noise1:
	python lsr.py train_data/noise_1.csv

noise2:
	python lsr.py train_data/noise_2.csv

noise3:
	python lsr.py train_data/noise_3.csv
