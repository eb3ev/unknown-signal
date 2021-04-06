all: basic adv noise

allPlot: basicPlot advPlot noisePlot

basic: basic1 basic2 basic3 basic4 basic5

basic1:
	python lsr.py train_data/basic_1.csv

basic2:
	python lsr.py train_data/basic_2.csv

basic3:
	python lsr.py train_data/basic_3.csv

basic4:
	python lsr.py train_data/basic_4.csv

basic5:
	python lsr.py train_data/basic_5.csv

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

basicPlot: basic1Plot basic2Plot basic3Plot basic4Plot basic5Plot

basic1Plot:
	python lsr.py train_data/basic_1.csv --plot

basic2Plot:
	python lsr.py train_data/basic_2.csv --plot

basic3Plot:
	python lsr.py train_data/basic_3.csv --plot

basic4Plot:
	python lsr.py train_data/basic_4.csv --plot

basic5Plot:
	python lsr.py train_data/basic_5.csv --plot

advPlot: adv1Plot adv2Plot adv3Plot

adv1Plot:
	python lsr.py train_data/adv_1.csv --plot

adv2Plot:
	python lsr.py train_data/adv_2.csv --plot

adv3Plot:
	python lsr.py train_data/adv_3.csv --plot

noisePlot: noise1Plot noise2Plot noise3Plot

noise1Plot:
	python lsr.py train_data/noise_1.csv --plot

noise2Plot:
	python lsr.py train_data/noise_2.csv --plot

noise3Plot:
	python lsr.py train_data/noise_3.csv --plot
