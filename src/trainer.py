import numpy as np
import torch
import torch.nn.functional as F

from .nt_xent import NTXentLoss


class Trainer:
    def __init__(self, model, optimizer, scheduler, batch_size, epochs, device, dataset, run_folder):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.dataset = dataset
        self.run_folder = run_folder

        self.nt_xent_criterion = NTXentLoss(
            device, batch_size, 0.5, True
        )

        self.model_name = 'model_{}.pth'.format(self.dataset)

    def _step(self, xis, xjs):
        ris, zis = self.model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = self.model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)

        return loss

    def _validate(self, val_loader):
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            for xis, xjs in val_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(xis, xjs)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        self.model.train()

        return valid_loss

    def train(self, train_loader, val_loader):
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.epochs):
            print(epoch_counter + 1, self.epochs)
            for xis, xjs in train_loader:
                self.optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(xis, xjs)

                loss.backward()

                self.optimizer.step()
                n_iter += 1

            valid_loss = self._validate(val_loader)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    self.model.state_dict(),
                    './results/{}/{}'.format(self.run_folder, self.model_name)
                )

            if epoch_counter >= 10:
                self.scheduler.step()

            valid_n_iter += 1

            if epoch_counter % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    './results/{}/checkpoints/'.format(self.run_folder) + str(epoch_counter) + '_' + self.model_name
                )
