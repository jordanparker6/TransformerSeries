make tensorboard:
	tensorboard --logdir=train/models/logs

make train:
	python3 src/main.py
