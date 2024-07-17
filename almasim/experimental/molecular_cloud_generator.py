#! /bin/env python3

# pip3 install turbustat

import matplotlib.pyplot as plt
import numpy as np
from turbustat.simulator.gen_field import make_extended


def return_random_molecular_cloud(imsize=256, powerlaw=None, ellip=None):
    """Creates a power-law distribution mimicking a molecular cloud structure"""
    import random

    if not powerlaw:
        powerlaw = random.random() * 3.0 + 1.5
    if not ellip:
        ellip = random.random() * 0.5 + 0.5
    theta = random.random() * 2 * 3.1415927

    print(
        f"Returning image with imsize={imsize} powerlawindex={powerlaw:.3}, theta={theta:.3} and ellip={ellip:.3}"
    )

    im = make_extended(
        imsize,
        powerlaw=powerlaw,
        theta=theta,
        ellip=ellip,
        randomseed=random.randrange(10000),
    )

    return im


if __name__ == "__main__":
    plt.imshow(return_random_molecular_cloud())
    plt.show()
