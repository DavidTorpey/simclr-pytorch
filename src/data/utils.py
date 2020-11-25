from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np


def get_train_validation_data_loaders(train_dataset, valid_size, batch_size, num_workers):
    # obtain training indices that will be used for validation
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, drop_last=True, shuffle=False)

    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler,
                              num_workers=num_workers, drop_last=True)

    return train_loader, valid_loader
