
k = 10
args = --no-cross-validation -v

all: basic adv noise

allPlot: basicPlot advPlot noisePlot

basic: basic1 basic2 basic3 basic4 basic5

basic1:
	python lsr.py train_data/basic_1.csv -k=$(k) $(args)

basic2:
	python lsr.py train_data/basic_2.csv -k=$(k) $(args)

basic3:
	python lsr.py train_data/basic_3.csv -k=$(k) $(args)

basic4:
	python lsr.py train_data/basic_4.csv -k=$(k) $(args)

basic5:
	python lsr.py train_data/basic_5.csv -k=$(k) $(args)

adv: adv1 adv2 adv3

adv1:
	python lsr.py train_data/adv_1.csv -k=$(k) $(args)

adv2:
	python lsr.py train_data/adv_2.csv -k=$(k) $(args)

adv3:
	python lsr.py train_data/adv_3.csv -k=$(k) $(args)

noise: noise1 noise2 noise3

noise1:
	python lsr.py train_data/noise_1.csv -k=$(k) $(args)

noise2:
	python lsr.py train_data/noise_2.csv -k=$(k) $(args)

noise3:
	python lsr.py train_data/noise_3.csv -k=$(k) $(args)

basicPlot: basic1Plot basic2Plot basic3Plot basic4Plot basic5Plot

basic1Plot:
	python lsr.py train_data/basic_1.csv -k=$(k) $(args) --plot

basic2Plot:
	python lsr.py train_data/basic_2.csv -k=$(k) $(args) --plot

basic3Plot:
	python lsr.py train_data/basic_3.csv -k=$(k) $(args) --plot

basic4Plot:
	python lsr.py train_data/basic_4.csv -k=$(k) $(args) --plot

basic5Plot:
	python lsr.py train_data/basic_5.csv -k=$(k) $(args) --plot

advPlot: adv1Plot adv2Plot adv3Plot

adv1Plot:
	python lsr.py train_data/adv_1.csv -k=$(k) $(args) --plot

adv2Plot:
	python lsr.py train_data/adv_2.csv -k=$(k) $(args) --plot

adv3Plot:
	python lsr.py train_data/adv_3.csv -k=$(k) $(args) --plot

noisePlot: noise1Plot noise2Plot noise3Plot

noise1Plot:
	python lsr.py train_data/noise_1.csv -k=$(k) $(args) --plot

noise2Plot:
	python lsr.py train_data/noise_2.csv -k=$(k) $(args) --plot

noise3Plot:
	python lsr.py train_data/noise_3.csv -k=$(k) $(args) --plot
