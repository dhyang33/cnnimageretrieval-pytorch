.PHONY: install
install:
	pip install -e .
	echo "You need to run the following command:"
	echo "source activate py3"

.PHONY: clean
clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

.PHONY: help
help:
	python ./cirtorch/examples/test.py -h

.PHONY: resnet
resnet:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'resnet101-gem' \
        --whitening 'retrieval-SfM-120k' --multiscale \
        --datasets 'oxford5k'

.PHONY: vgg
vgg:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'vgg16-gem' \
        --whitening 'retrieval-SfM-120k' --multiscale \
        --datasets 'oxford5k'

.PHONY: tuned-resnet
tuned-resnet:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path 'retrievalSfM120k-resnet101-gem' \
		--whitening 'retrieval-SfM-120k' --multiscale \
		--datasets 'oxford5k'

.PHONY: tuned-vgg
tuned-vgg:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path 'retrievalSfM120k-vgg16-gem' \
		--whitening 'retrieval-SfM-120k' --multiscale \
		--datasets 'oxford5k'
