import pathlib
import sys

import torch
import yaml
from torchvision import datasets
import numpy as np

from .model.backbone import ResNetSimCLR
from .data.utils import get_train_validation_data_loaders
from .trainer import Trainer
from .data.dataset import CustomDataset


def main():
    run_folder = sys.argv[-2]

    path = pathlib.Path('./results/{}/checkpoints'.format(run_folder))
    path.mkdir(exist_ok=True, parents=True)

    config = yaml.load(open(sys.argv[-1], "r"), Loader=yaml.FullLoader)
    dataset = config['data']['dataset']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")
    print('Using dataset:', dataset)

    if dataset == 'stl10':
        train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True)
        trainset = train_dataset.data
        trainset = np.swapaxes(np.swapaxes(trainset, 1, 2), 2, 3)
    elif dataset == 'cifar10':
        train_dataset = datasets.CIFAR10('./data', train=True, download=True)
        trainset = train_dataset.data
    elif dataset == 'svhn':
        train_dataset = datasets.SVHN('./data', split='train', download=True)
        trainset = train_dataset.data
        trainset = np.swapaxes(np.swapaxes(trainset, 1, 2), 2, 3)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100('./data', train=True, download=True)
        trainset = train_dataset.data

    train_dataset = CustomDataset(trainset, config['data'])

    model = ResNetSimCLR(**config['network']).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config['optimizer']['lr']),
        weight_decay=float(config['optimizer']['weight_decay'])
    )

    batch_size = config['trainer']['batch_size']
    epochs = config['trainer']['epochs']

    train_loader, valid_loader = get_train_validation_data_loaders(
        train_dataset, 0.05, batch_size,
        config['trainer']['num_workers']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )

    trainer = Trainer(
        model, optimizer, scheduler, batch_size,
        epochs, device, dataset, run_folder
    )

    trainer.train(train_loader, valid_loader)


if __name__ == '__main__':
    main()
