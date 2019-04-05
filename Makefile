export PYTHONPATH := ""

.PHONY: install
install:
	echo "You need to run all of the following commands:"
	echo "source activate py3"
	echo "export PYTHONPATH=\"${PYTHONPATH}\""
	pip install -Ue .

.PHONY: clean
clean:
	rm -rf ./logs
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

.PHONY: help
help:
	python ./cirtorch/examples/test.py -h

.PHONY: run-benchmarks
run-benchmarks: clean
	-mkdir ./logs
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'resnet101-gem' --multiscale --datasets 'scores' > ./logs/resnet-gem.log
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'vgg16-gem' --multiscale --datasets 'scores' > ./logs/vgg-gem.log
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path 'retrievalSfM120k-resnet101-gem' --multiscale --datasets 'scores' > ./logs/pretuned-resnet-gem.log --whitening 'retrieval-SfM-120k'
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path 'retrievalSfM120k-vgg16-gem' --multiscale --datasets 'scores' > ./logs/pretuned-vgg-gem.log --whitening 'retrieval-SfM-120k'
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'resnet101-rmac' --multiscale --datasets 'scores' > ./logs/resnet-rmac.log
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'vgg16-rmac' --multiscale --datasets 'scores' > ./logs/vgg-rmac.log
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'resnet101-mac' --multiscale --datasets 'scores' > ./logs/resnet-mac.log
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'vgg16-mac' --multiscale --datasets 'scores' > ./logs/vgg-mac.log
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'resnet101-spoc' --multiscale --datasets 'scores' > ./logs/resnet-spoc.log
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'vgg16-spoc' --multiscale --datasets 'scores' > ./logs/vgg-spoc.log

.PHONY: resnet-gem
resnet-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'resnet101-gem' --multiscale --datasets 'scores'

.PHONY: vgg-gem
vgg-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'vgg16-gem' --multiscale --datasets 'scores'

.PHONY: pretuned-resnet-gem
pretuned-resnet-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path 'retrievalSfM120k-resnet101-gem' --multiscale --datasets 'scores' --whitening 'retrieval-SfM-120k'

.PHONY: pretuned-vgg-gem
pretuned-vgg-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path 'retrievalSfM120k-vgg16-gem' --multiscale --datasets 'scores' --whitening 'retrieval-SfM-120k'

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

.PHONY: tuned-vgg-gem
tuned-vgg-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path ./weights/scores_vgg16_gem_whiten_contrastive_m0.85_adam_lr1.0e-06_wd1.0e-04_nnum5_qsize2000_psize20000_bsize4_imsize362/model_best.pth.tar --multiscale --datasets 'scores' --whitening 'scores'

.PHONY: train-example-vgg-gem
train-example-vgg-gem:
	python ./cirtorch/examples/train.py ./weights --gpu-id '0' --training-dataset 'retrieval-SfM-120k' --test-datasets 'roxford5k,rparis6k' --arch 'vgg16' --pool 'gem' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 1e-6 --neg-num 5 --query-size=2000 --pool-size=20000 --batch-size 5 --image-size 362 --whitening

.PHONY: train-vgg-gem
train-vgg-gem:
	python ./cirtorch/examples/train.py ./weights --gpu-id '0' --training-dataset 'scores' --test-datasets 'scores' --arch 'vgg16' --pool 'gem' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 1e-6 --neg-num 5 --query-size=2000 --pool-size=20000 --batch-size 4 --image-size 362 --whitening --test-whiten 'scores'

.PHONY: train-vgg-rmac
train-vgg-rmac:
	python ./cirtorch/examples/train.py ./weights --gpu-id '0' --training-dataset 'scores' --test-datasets 'scores' --arch 'vgg16' --pool 'rmac' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 1e-6 --neg-num 5 --query-size=2000 --pool-size=20000 --batch-size 4 --image-size 362 --whitening --test-whiten 'scores'

.PHONY: train-resnet-gem
train-resnet-gem:
	python ./cirtorch/examples/train.py ./weights --gpu-id '0' --training-dataset 'scores' --test-datasets 'scores' --arch 'resnet101' --pool 'gem' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 1e-6 --neg-num 5 --query-size=2000 --pool-size=20000 --batch-size 4 --image-size 362 --whitening --test-whiten 'scores'

.PHONY: train-resnet-rmac
train-resnet-rmac:
	python ./cirtorch/examples/train.py ./weights --gpu-id '0' --training-dataset 'scores' --test-datasets 'scores' --arch 'resnet101' --pool 'rmac' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 1e-6 --neg-num 5 --query-size=2000 --pool-size=20000 --batch-size 4 --image-size 362 --whitening --test-whiten 'scores'
