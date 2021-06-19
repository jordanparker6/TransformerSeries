make tensorboard:
	tensorboard --logdir=train/models

make training:
	python3 src/main.py

tests:
	pytest src