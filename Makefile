export PYTHONPATH := ""

.PHONY: install
install:
	echo "You need to run all of the following commands:"
	echo "source activate py3"
	echo "export PYTHONPATH=\"${PYTHONPATH}\""
	pip install -e .

.PHONY: clean
clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

.PHONY: help
help:
	python ./cirtorch/examples/test.py -h

.PHONY: resnet-gem
resnet-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'resnet101-gem' --multiscale --datasets 'scores'

.PHONY: vgg-gem
vgg-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'vgg16-gem' --multiscale --datasets 'scores'

.PHONY: tuned-resnet-gem
tuned-resnet-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path 'retrievalSfM120k-resnet101-gem' --multiscale --datasets 'scores'

.PHONY: tuned-vgg-gem
tuned-vgg-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path 'retrievalSfM120k-vgg16-gem' --multiscale --datasets 'scores'

.PHONY: resnet-rmac
resnet-rmac:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'resnet101-rmac' --multiscale --datasets 'scores'

.PHONY: vgg-rmac
vgg-rmac:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'vgg16-rmac' --multiscale --datasets 'scores'

.PHONY: resnet-mac
resnet-mac:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'resnet101-mac' --multiscale --datasets 'scores'

.PHONY: vgg-mac
vgg-mac:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'vgg16-mac' --multiscale --datasets 'scores'

.PHONY: resnet-spoc
resnet-spoc:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'resnet101-spoc' --multiscale --datasets 'scores'

.PHONY: vgg-spoc
vgg-spoc:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'vgg16-spoc' --multiscale --datasets 'scores'
