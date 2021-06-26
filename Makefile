make install:
	pip3 install -r requirements.txt

make tensorboard:
	tensorboard --logdir=train/models

make training:
	python3 src/main.py

tests:
	pytest src

profile:
	 python -m torch.utils.bottleneck src/main.py