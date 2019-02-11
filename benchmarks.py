import os

import torch
import numpy as np
from torch.utils.model_zoo import load_url
from torchvision import transforms
from torch.autograd import Variable

from cirtorch.utils.general import get_data_root
from cirtorch.networks.imageretrievalnet import (
    init_network,
    extract_vectors,
    extract_ss,
    extract_ms,
)
from cirtorch.datasets.genericdataset import ImagesFromDataList
from cirtorch.examples.test import (
    load_network,
    load_offtheshelf,
)

from score_retrieval.data import indices_with_label


def vectors_from_images(net, images, transform, ms=[1], msp=1, print_freq=10, setup_network=True):
    """Extract vectors from images given as a pytorch array."""
    # moving network to gpu and eval mode
    if setup_network:
        net.cuda()
        net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromDataList(images=images, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    # extracting vectors
    vecs = torch.zeros(net.meta['outputdim'], len(images))
    for i, input in enumerate(loader):
        input_var = Variable(input.cuda())

        if len(ms) == 1:
            vecs[:, i] = extract_ss(net, input_var)
        else:
            vecs[:, i] = extract_ms(net, input_var, ms, msp)

        if (i+1) % print_freq == 0 or (i+1) == len(images):
            print('{}/{}...'.format((i+1), len(images)), end='')
    print('done')
    return vecs


def call_benchmark(
    # must pass one of images or paths
    images=None,
    paths=None,

    network="retrievalSfM120k-resnet101-gem",
    offtheshelf=False,
    image_size=1024,
    gpu=False,
):
    """Run the given network on the given data and return vectors for it."""
    # load network
    if offtheshelf:
        net = load_offtheshelf(network)
    else:
        net = load_network(network)

    # setting up the multi-scale parameters
    ms = [1]
    msp = 1

    # moving network to gpu and eval mode
    if gpu:
        net.cuda()
    net.eval()

    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # process the given data
    if images is not None:
        vecs = vectors_from_images(net, np.asarray(images), transform, ms=ms, msp=msp, setup_network=False)
    else:
        vecs = extract_vectors(net, paths, image_size, transform, ms=ms, msp=msp, setup_network=False)

    # convert to numpy
    return vecs.numpy()
