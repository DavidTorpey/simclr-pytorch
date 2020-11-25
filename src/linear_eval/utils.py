import torch
import numpy as np


def get_features_from_encoder(encoder, loader, device):
    x_train = []
    y_train = []

    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        feature_vector = encoder(x)
        fvs = feature_vector.cpu().detach().numpy()
        ys = y.cpu().detach().numpy()

        for e1, e2 in zip(fvs, ys):
            x_train.append(e1.ravel())
            y_train.append(e2)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train


def get_numpy_data(encoder, train_loader, val_loader, test_loader, device):
    encoder.eval()

    x_train, y_train = get_features_from_encoder(encoder, train_loader, device)
    x_val, y_val = get_features_from_encoder(encoder, val_loader, device)
    x_test, y_test = get_features_from_encoder(encoder, test_loader, device)

    return x_train, y_train, x_val, y_val, x_test, y_test
