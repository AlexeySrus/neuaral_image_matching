import torch
import argparse
import os
from PIL import Image
import yaml
import numpy as np
import torch.nn.functional as F
from image_matching.model.model import Model, get_last_epoch_weights_path
from image_matching.utils.callbacks import (SaveModelPerEpoch, VisPlot,
                                      SaveOptimizerPerEpoch,
                                        VisImageForMatcher)
from image_matching.utils.dataset_generator import TransformFramesDataset
from image_matching.architectures.match_model import MatchModel
from image_matching.utils.losses import l2
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Images matching train script')
    parser.add_argument('--config', required=False, type=str,
                          default='../configuration/train_config.yml',
                          help='Path to configuration yml file.'
                        )
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    return parser.parse_args()


def load_images_from_path(path):
    names = os.listdir(path)
    return [
        np.array(Image.open(path + '/' + name).convert('RGB'))
        for name in names
    ]


def main():
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    batch_size = config['train']['batch_size']
    n_jobs = config['train']['number_of_processes']
    epochs = config['train']['epochs']

    model = Model(
        MatchModel(),
        device
    )

    callbacks = []

    callbacks.append(SaveModelPerEpoch(
        os.path.join(
            os.path.dirname(__file__),
            config['train']['save']['model']
        ),
        config['train']['save']['every']
    ))

    callbacks.append(SaveOptimizerPerEpoch(
        os.path.join(
            os.path.dirname(__file__),
            config['train']['save']['model']
        ),
        config['train']['save']['every']
    ))

    if config['visualization']['use_visdom']:
        plots = VisPlot(
            'Images matching train',
            server=config['visualization']['visdom_server'],
            port=config['visualization']['visdom_port']
        )

        plots.register_scatterplot('train loss per_batch', 'Batch number',
                                   'Loss',
                                   [
                                       'loss'
                                   ])

        callbacks.append(plots)

        callbacks.append(
            VisImageForMatcher(
                'Image visualisation',
                config['visualization']['visdom_server'],
                config['visualization']['visdom_port'],
                config['visualization']['image']['every'],
                scale=config['visualization']['image']['scale']
            )
        )

    model.set_callbacks(callbacks)

    start_epoch = 0
    optimizer = torch.optim.Adam(
        model.model.parameters(),
        lr=config['train']['lr']
    )

    if config['train']['load']:
        weight_path, optim_path, start_epoch = get_last_epoch_weights_path(
            os.path.join(
                os.path.dirname(__file__),
                config['train']['save']['model']
            ),
            print
        )

        if weight_path is not None:
            model.load(weight_path)
            optimizer.load_state_dict(torch.load(optim_path,
                                                 map_location='cpu'))

    train_data = DataLoader(
        TransformFramesDataset(
            images_path=config['dataset']['images_path'],
            shape=config['dataset']['shape'],
            transform_rect_size=config[
                'dataset'
            ]['transformation_rectangle_size'],
            transform_deviate=config['dataset']['transformation_deviate'],
        ),
        batch_size=batch_size,
        num_workers=n_jobs
    )

    model.fit(
        train_data,
        optimizer,
        epochs,
        l2,
        init_start_epoch=start_epoch + 1,
        validation_loader=None
    )


if __name__ == '__main__':
    main()
