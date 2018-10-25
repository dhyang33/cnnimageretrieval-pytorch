.PHONY: install
install:
	echo "You need to run the following command:"
	echo "source activate py3"
	pip install -e .

.PHONY: clean
clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

.PHONY: help
help:
	python ./cirtorch/examples/test.py -h

.PHONY: resnet
resnet:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'resnet101-gem' --multiscale --datasets 'scores,rscores'

.PHONY: vgg
vgg:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'vgg16-gem' --multiscale --datasets 'scores,rscores'

.PHONY: tuned-resnet
tuned-resnet:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path 'retrievalSfM120k-resnet101-gem' --multiscale --datasets 'scores,rscores'

.PHONY: tuned-vgg
tuned-vgg:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path 'retrievalSfM120k-vgg16-gem' --multiscale --datasets 'scores,rscores'
