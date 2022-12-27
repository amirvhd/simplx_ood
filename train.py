import os
import argparse

from Dataloader.cifar_datamodule import cifar10_module
from Trainer.cifar10_trainer import classifier_module
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

RANDOM_SEED = 42

pl.seed_everything(RANDOM_SEED)


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-5,
                        help='weight decay')
    parser.add_argument('--n_cls', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs')
    parser.add_argument('--data_folder', type=str, default="./data", help='path to custom dataset')
    parser.add_argument('--n_workers', type=int, default=16,
                        help='number of workers')
    parser.add_argument('--n_freeze_layers', type=int, default=0,
                        help='number of layers that are frozen')
    parser.add_argument('--sn', action='store_true',
                        help='using spectral normalize layers')
    parser.add_argument('--bn', action='store_true',
                        help='using spectral normalize batch normalization')
    opt = parser.parse_args()

    # set the path according to the environment

    opt.model_path = './model'
    if not os.path.isdir(opt.model_path):
        os.makedirs(opt.model_path)
    return opt


def main():
    opt = parse_option()
    # import dataset
    data_module = cifar10_module(data_dir=opt.data_folder,
                                 batch_size=opt.batch_size,
                                 n_workers=opt.n_workers
                                 )
    # import trainer
    model = classifier_module(
        n_classes=opt.n_cls, lr=opt.lr, wd=opt.wd, n_layers=opt.n_freeze_layers, bn=opt.bn, sn=opt.sn
    )

    # callbacks definitions
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.model_path,
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_acc",
        mode="max"
    )
    logger = TensorBoardLogger(save_dir=opt.log_dir, name="classification")
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1
    # training
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, TQDMProgressBar(refresh_rate=30)],
        max_epochs=opt.epochs,
        strategy="ddp_find_unused_parameters_false",
        gpus=n_gpus
    )
    trainer.fit(model, data_module)

    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()
