import torchvision
from torch.utils.data import random_split
from torchvision import transforms, datasets


def get_datasets(dataset, image_size, val_p=0.1):
    train_transforms = torchvision.transforms.Compose([
        transforms.ToTensor()
    ])

    test_transforms = torchvision.transforms.Compose([
        transforms.ToTensor()
    ])

    if dataset == 'stl10':
        train_dataset = datasets.STL10('./data', split='train', download=True,
                                       transform=train_transforms)

        test_dataset = datasets.STL10('./data', split='test', download=True,
                                      transform=test_transforms)
    elif dataset == 'cifar10':
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)
    elif dataset == 'svhn':
        train_dataset = datasets.SVHN('./data', split='train', download=True, transform=train_transforms)
        test_dataset = datasets.SVHN('./data', split='test', download=True, transform=test_transforms)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR100('./data', train=False, download=True, transform=test_transforms)

    num_val = int(len(train_dataset) * val_p)
    num_train = len(train_dataset) - num_val
    train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])

    return train_dataset, val_dataset, test_dataset