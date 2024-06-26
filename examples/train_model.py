import argparse
from pathlib import Path

from aitrainer import Trainer, init_distritubed, load_config, log_info

from .data import get_mnist_data_loader
from .hparams import MLPHparams
from .models import MLP, log_fn, log_valid_fn, train_step_fn, valid_fn


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams_fp', type=Path, default='examples/hparams.json')
    return parser.parse_args()


def train(hparams_fp: Path):
    init_distritubed()
    hparams = MLPHparams(**load_config(hparams_fp))
    model = MLP(hparams)
    trainer = Trainer(
        hparams=hparams,
        model_dict={'model': model},
        train_step_fn=train_step_fn,
        valid_fn=valid_fn,
        train_log_fn=log_fn,
        valid_log_fn=log_valid_fn,
    )

    train_dl = get_mnist_data_loader(hparams.batch_size, is_train=True)
    valid_dl = get_mnist_data_loader(hparams.batch_size, is_train=False)

    log_info('Starting training')
    trainer.train(train_dl, valid_dl)
    log_info('Training complete')


if __name__ == '__main__':
    params = get_params()
    train(params.hparams_fp)
