make tensorboard:
	tensorboard --logdir=train/models

make train:
	python3 src/main.py
