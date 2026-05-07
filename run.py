from cifar10_resnet20 import model, testset, testloader
from attack import (
    fgsm, ifgsm, mifgsm,
    projected_gradient_descent, frank_wolfe,
    lbfgs
)
from utils import get_correct_indices, attack_success_rate
from torch.utils.data import Subset, DataLoader
import numpy as np


if __name__ == "__main__":
    indices = get_correct_indices(model, testloader, 512)
    subset = Subset(testset, indices)
    loader = DataLoader(subset, batch_size=128, shuffle=False)

    e = 0.01
    p = 1
    N = 10
    attack = mifgsm

    success_rate, successful_indices = attack_success_rate(loader, model, attack, e=e, p=p, N=N)
