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


def vectors_from_images(net, images, transform, ms=[1], msp=1, print_freq=10, setup_network=True, gpu=True):
    """Extract vectors from images given as a pytorch array."""
    # moving network to gpu and eval mode
    if setup_network:
        if gpu:
            net.cuda()
        net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromDataList(images=images, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    # extracting vectors
    vecs = torch.zeros(net.meta['outputdim'], len(images))
    for i, input_data in enumerate(loader):
        if gpu:
            input_data = input_data.cuda()
        input_var = Variable(input_data)

        # fix tensor shape
        if len(input_var.shape) > 4 and input_var.shape[0] == 1:
            input_var = input_var[0]

        if len(ms) == 1:
            vecs[:, i] = extract_ss(net, input_var)
        else:
            vecs[:, i] = extract_ms(net, input_var, ms, msp)

        if (i+1) % print_freq == 0 or (i+1) == len(images):
            print('{}/{}...'.format((i+1), len(images)), end='')
    print('done')
    return vecs


LOADED_NETWORKS = {}


def call_benchmark(
    # must pass one of images or paths
    images=None,
    paths=None,

    network="retrievalSfM120k-vgg16-gem",
    offtheshelf=False,
    image_size=1024,
    gpu=True,
):
    """Run the given network on the given data and return vectors for it."""
    net_key = (network, offtheshelf, gpu)

    if net_key in LOADED_NETWORKS:
        net = LOADED_NETWORKS[net_key]

    else:
        # load network
        if offtheshelf:
            net = load_offtheshelf(network)
        else:
            net = load_network(network)

        # moving network to gpu and eval mode
        if gpu:
            net.cuda()
        net.eval()

        # store network in memo dict
        LOADED_NETWORKS[net_key] = net

    # setting up the multi-scale parameters
    ms = [1]
    msp = 1

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
        images = np.asarray(images)
        print("images.shape =", images.shape)
        vecs = vectors_from_images(net, images, transform, ms=ms, msp=msp, setup_network=False, gpu=gpu)
    else:
        vecs = extract_vectors(net, paths, image_size, transform, ms=ms, msp=msp, setup_network=False, gpu=gpu)

    # convert to numpy
    np_vecs = vecs.numpy().T
    print("np_vecs.shape =", np_vecs.shape)
    return np_vecs
