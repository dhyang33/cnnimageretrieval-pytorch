import os

import torch
import numpy as np
from torch.utils.model_zoo import load_url
from torchvision import transforms

from cirtorch.utils.general import get_data_root
from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.examples.test import (
    load_network,
    load_offtheshelf,
)

from score_retrieval.data import indices_with_label


def call_benchmark(
    database_images,
    database_labels,
    query_images,
    query_labels,
    network,
    offtheshelf=True,
    image_size=1024
):
    """Run the given network on the given data and return ranks, gnd."""
    if offtheshelf:
        net = load_offtheshelf(network)
    else:
        net = load_network(network)

    # setting up the multi-scale parameters
    ms = [1]
    msp = 1

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # process the given data
    gnd = [{"ok": indices_with_label(label, database_labels), "junk": []} for label in query_labels]

    print('>> processing database images...')
    vecs = extract_vectors(net, database_images, image_size, transform, ms=ms, msp=msp)
    print('>> processing query images...')
    qvecs = extract_vectors(net, query_images, image_size, transform, ms=ms, msp=msp)

    # convert to numpy
    vecs = vecs.numpy()
    qvecs = qvecs.numpy()

    # search, rank, and print
    scores = np.dot(vecs.T, qvecs)
    ranks = np.argsort(-scores, axis=0)

    return ranks, gnd
