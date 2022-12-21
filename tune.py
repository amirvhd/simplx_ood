import os
import argparse

from Dataloader.cifar_datamodule import cifar10_module
from Trainer.cifar10_trainer import classifier_module
from ray import air, tune, init
from ray.tune import CLIReporter
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 42

pl.seed_everything(RANDOM_SEED)


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4,
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
    opt.model_path = '/dss/dssmcmlfs01/pn69za/pn69za-dss-0002/ra49bid2//saved_models/BERD/'
    opt.log_dir = os.path.join(opt.data_folder, "Berd_logging/image_classification")
    if not os.path.isdir(opt.model_path):
        os.makedirs(opt.model_path)
    return opt


def train(config, dirpath, data_dir,
          batch_size,
          n_workers,
          save_dir, n_classes, n_layers=1,
          max_epochs=1):
    data_module = cifar10_module(data_dir,
                                 batch_size,
                                 n_workers
                                 )
    # import trainer
    model = classifier_module(
        n_classes, lr=config["lr"], wd=config["wd"], n_layers=n_layers
    )

    # callbacks definitions
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    logger = TensorBoardLogger(save_dir, name="classification")
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
    tuning_callback = TuneReportCallback(
        {
            "loss": "val_loss",
            "acc": "val_acc"
        },
        on="validation_end")
    # training
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, TQDMProgressBar(refresh_rate=30), tuning_callback],
        max_epochs=max_epochs,
    )
    trainer.fit(model, data_module)


def main():
    opt = parse_option()
    # import dataset
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "wd": tune.loguniform(1e-7, 1e-4),
        "drop": tune.uniform(0, 4e-1)
    }
    init(ignore_reinit_error=True, num_cpus=16)
    # Bayesian optimization
    algo = TuneBOHB(metric="mean_loss", mode="min")
    bohb = HyperBandForBOHB(time_attr="training_iteration",
                            max_t=10,
                            reduction_factor=2,
                            stop_last_trials=True
                            )
    # Asynchronous HyperBand
    scheduler = ASHAScheduler(
        max_t=opt.epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["lr","wd","drop"],
        metric_columns=["loss", "acc"])
    train_fn_with_parameters = tune.with_parameters(train, data_dir=opt.data_folder,
                                                    batch_size=opt.batch_size,
                                                    n_workers=opt.n_workers,
                                                    save_dir=opt.log_dir,
                                                    n_classes=opt.n_cls,
                                                    dirpath=opt.model_path,
                                                    n_layers=opt.n_freeze_layers,
                                                    max_epochs=opt.epochs)
    resources_per_trial = {"gpu": 2}
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            reuse_actors=True,
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=20,
            # search_alg=algo,
        ),
        run_config=air.RunConfig(
            name="main",
            progress_reporter=reporter,
        ),
        param_space=config,

    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)
    # print("Best hyperparameters found were: ", results.get_dataframe())
    # trainer.test(model=best_model, datamodule=data_module,
    #              # ckpt_path='best'
    #              )
    plot = sns.scatterplot(x="config/lr", y="config/wd", data=results.get_dataframe(), size="acc",
                           legend=False)
    plot.set(xscale="log")
    plot.set(yscale="log")
    plt.savefig('Asynchronous_HyperBand.png')
    # plt.savefig('bohb.png')

if __name__ == '__main__':
    main()
