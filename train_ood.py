import os
import argparse

from Dataloader.cifar_datamodule import cifar10_module
from Trainer.cifar10_trainer import classifier_module
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from ood_metrics import calc_metrics


def calc_auroc(ind_score: np.ndarray, ood_score: np.ndarray) -> dict:
    labels = [1] * len(ind_score) + [0] * len(ood_score)
    scores = np.hstack([ind_score, ood_score])

    metric_dict = calc_metrics(scores, labels)
    # fpr, tpr, _ = roc_curve(labels, scores)

    metric_dict_transformed = {
        "AUROC": 100 * metric_dict["auroc"],
        #    "TNR at TPR 95%": 100 * (1 - metric_dict["fpr_at_95_tpr"]),
        #   "Detection Acc.": 100 * 0.5 * (tpr + 1 - fpr).max(),
    }
    return metric_dict_transformed


RANDOM_SEED = 42

pl.seed_everything(RANDOM_SEED)


# Search done. Best Val accuracy = 0.8193999528884888
# Best Config: {'lr': 0.001, 'wd': 1e-05}
# [({'lr': 0.0001, 'wd': 0.0001}, tensor(0.7402)), ({'lr': 0.0001, 'wd': 1e-05}, tensor(0.7498)), ({'lr': 0.0001, 'wd': 1e-06}, tensor(0.7409)), ({'lr': 0.001, 'wd': 0.0001}, tensor(0.8103)), ({'lr': 0.001, 'wd': 1e-05}, tensor(0.8194)), ({'lr': 0.001, 'wd': 1e-06}, tensor(0.8160)), ({'lr': 0.01, 'wd': 0.0001}, tensor(0.4550)), ({'lr': 0.01, 'wd': 1e-05}, tensor(0.4594)), ({'lr': 0.01, 'wd': 1e-06}, tensor(0.3565)), ({'lr': 0.1, 'wd': 0.0001}, tensor(0.1001)), ({'lr': 0.1, 'wd': 1e-05}, tensor(0.1001)), ({'lr': 0.1, 'wd': 1e-06}, tensor(0.0969))]

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
    parser.add_argument('--data_folder', type=str, default="~/DATA2", help='path to custom dataset')
    parser.add_argument('--n_workers', type=int, default=16,
                        help='number of workers')
    parser.add_argument('--n_freeze_layers', type=int, default=0,
                        help='number of layers that are frozen')

    opt = parser.parse_args()

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '~/DATA2'
    opt.model_path = '/dss/dssmcmlfs01/pn69za/pn69za-dss-0002/ra49bid2/saved_models/BERD/'
    opt.log_dir = os.path.join(opt.data_folder, "Berd_logging/image_classification")
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
        n_classes=opt.n_cls, lr=opt.lr, wd=opt.wd, n_layers=opt.n_freeze_layers
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
        # n_gpus = 2
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1
    # training
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=30)],
        max_epochs=opt.epochs,
        strategy="ddp_find_unused_parameters_false",
        gpus=n_gpus
    )
    # trainer.fit(model, data_module)

    ind = trainer.test(model, datamodule=data_module,
                       ckpt_path=os.path.join(opt.model_path, 'best-checkpoint.ckpt')
                       )
    ood = trainer.predict(model, datamodule=data_module,
                       ckpt_path=os.path.join(opt.model_path, 'best-checkpoint.ckpt')
                       )
    print(ind)
    # calc_auroc(ind, ood)


if __name__ == '__main__':
    main()
