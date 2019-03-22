import torch
import tqdm
import os
import re
from image_matching.utils.losses import l2
from image_matching.utils.losses import acc as acc_function
from image_matching.utils.tensor_utils import flatten
from image_matching.utils.tensor_utils import crop_batch_by_center


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Model:
    def __init__(self, net, _device='cpu', callbacks_list=None):
        self.device = torch.device('cpu' if _device == 'cpu' else 'cuda')
        self.model = net.to(self.device)
        self.callbacks = [] if callbacks_list is None else callbacks_list
        self.last_n = 0
        self.last_optimiser_state = None

    def fit(self,
            train_loader,
            optimizer,
            epochs=1,
            loss_function=l2,
            validation_loader=None,
            verbose=False,
            init_start_epoch=1,
            acc_f=acc_function):
        """
        Model train method
        Args:
            train_loader: DataLoader
            optimizer: optimizer from torch.optim with initialized parameters
            or tuple of (optimizer, scheduler)
            epochs: epochs count
            loss_function: Loss function
            validation_loader: DataLoader
            verbose: print evaluate validation prediction
            init_start_epoch: start epochs number
        Returns:
        """
        scheduler = None
        if type(optimizer) is tuple:
            scheduler = optimizer[1]
            optimizer = optimizer[0]

        for epoch in range(init_start_epoch, epochs + 1):
            self.model.train()

            batches_count = len(train_loader)
            avg_epoch_loss = 0
            avg_epoch_acc = 0

            if scheduler is not None:
                scheduler.step(epoch)

            self.last_n = epoch

            with tqdm.tqdm(total=batches_count) as pbar:
                for i, batch in enumerate(train_loader):
                    self.last_optimiser_state = optimizer.state_dict()

                    original_img = batch[0].to(self.device)
                    transformed_img = batch[1].to(self.device)
                    y_true = batch[2].to(self.device)

                    optimizer.zero_grad()
                    y_pred = self.model(original_img, transformed_img)

                    loss = loss_function(y_pred, y_true)
                    loss.backward()
                    optimizer.step()

                    acc = 0
                    # acc = acc_f(
                    #     flatten(y_pred),
                    #     flatten(crop_batch_by_center(y_true, y_pred.shape))
                    # )

                    pbar.postfix = \
                        'Epoch: {}/{}, loss: {:.8f}, acc: {:.8f}, lr: {:.8f}'.format(
                            epoch,
                            epochs,
                            loss.item() / train_loader.batch_size,
                            acc,
                            get_lr(optimizer)
                        )
                    avg_epoch_loss += \
                        loss.item() / train_loader.batch_size / batches_count

                    avg_epoch_acc += acc

                    for cb in self.callbacks:
                        cb.per_batch({
                            'model': self,
                            'loss': loss.item() / train_loader.batch_size,
                            'n': (epoch - 1)*batches_count + i + 1,
                            'original': original_img,
                            'transformed': transformed_img,
                            'y_pred': y_pred,
                            'y_true': y_true,
                            'acc': acc
                        })

                    pbar.update(1)

            test_loss = None
            test_acc = None

            for cb in self.callbacks:
                cb.per_epoch({
                    'model': self,
                    'loss': avg_epoch_loss,
                    'val loss': test_loss,
                    'n': epoch,
                    'optimize_state': optimizer.state_dict(),
                    'acc': avg_epoch_acc,
                    'val acc': test_acc
                })

    def evaluate(self,
                 test_loader,
                 loss_function=l2,
                 verbose=False,
                 acc_f=acc_function):
        """
        Test model
        Args:
            test_loader: DataLoader
            loss_function: loss function
            verbose: print progress

        Returns:

        """
        pass

    def predict(self,
                image):
        pass

    def set_callbacks(self, callbacks_list):
        self.callbacks = callbacks_list

    def save(self, path):
        torch.save(self.model.cpu().state_dict(), path)
        self.model = self.model.to(self.device)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.eval()
        self.model = self.model.to(self.device)

    def __del__(self):
        for cb in self.callbacks:
            cb.early_stopping(
                {
                    'model': self,
                    'n': self.last_n,
                    'optimize_state': self.last_optimiser_state
                }
            )


def get_last_epoch_weights_path(checkpoints_dir, log=None):
    """
    Get last epochs weights from target folder
    Args:
        checkpoints_dir: target folder
        log: logging, default standard print
    Returns:
        (
            path to current weights file,
            path to current optimiser file,
            current epoch number
        )
    """
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        return None, None, 0

    weights_files_list = [
        matching_f.group()
        for matching_f in map(
            lambda x: re.match('model-\d+.trh', x),
            os.listdir(checkpoints_dir)
        ) if matching_f if not None
    ]

    if len(weights_files_list) == 0:
        return None, None, 0

    weights_files_list.sort(key=lambda x: -int(x.split('-')[1].split('.')[0]))

    if log is not None:
        log('LOAD MODEL PATH: {}'.format(
            os.path.join(checkpoints_dir, weights_files_list[0])
        ))

    n = int(
        weights_files_list[0].split('-')[1].split('.')[0]
    )

    return os.path.join(checkpoints_dir,
                        weights_files_list[0]
                        ), \
           os.path.join(checkpoints_dir, 'optimize_state-{}.trh'.format(n)), n