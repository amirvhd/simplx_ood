import pickle as pkl
import matplotlib.pyplot as plt
import os
import argparse

import numpy

from Dataloader.cifar_datamodule import cifar10_module
from Trainer.cifar10_trainer import classifier_module
import torch

from ood_metrics import calc_metrics


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


def calc_auroc(ind_score: numpy.ndarray, ood_score: numpy.ndarray) -> dict:
    labels = [1] * len(ind_score) + [0] * len(ood_score)
    scores = numpy.hstack([ind_score, ood_score])

    metric_dict = calc_metrics(scores, labels)
    # fpr, tpr, _ = roc_curve(labels, scores)

    metric_dict_transformed = {
        "AUROC": 100 * metric_dict["auroc"],
        #    "TNR at TPR 95%": 100 * (1 - metric_dict["fpr_at_95_tpr"]),
        #   "Detection Acc.": 100 * 0.5 * (tpr + 1 - fpr).max(),
    }
    return metric_dict_transformed


def main():
    with open('./experiments/results/cifar/outlier/simplex_cv0.pkl', 'rb') as f:
        data_base = pkl.load(f)
    with open('./experiments/results/cifar/outlier_sn/simplex_cv0.pkl', 'rb') as f:
        data_sn = pkl.load(f)
    pred_base = data_base.latent_approx()
    error_base = ((data_base.test_latent_reps - pred_base) ** 2).sum(1).cpu().numpy()
    pred_sn = data_sn.latent_approx()
    error_sn = ((data_sn.test_latent_reps - pred_sn) ** 2).sum(1).cpu().numpy()

    mean_base = numpy.mean(error_base[:10000])
    std_base = numpy.std(error_base[:10000])
    dist_in_base = torch.distributions.normal.Normal(loc=mean_base, scale=std_base)
    mean_sn = numpy.mean(error_base[:10000])
    std_sn = numpy.std(error_base[:10000])
    dist_in_sn = torch.distributions.normal.Normal(loc=mean_sn, scale=std_sn)
    pdf_cifar10_base = torch.zeros(10000)
    pdf_cifar100_base = torch.zeros(10000)
    pdf_cifar10_sn = torch.zeros(10000)
    pdf_cifar100_sn = torch.zeros(10000)
    for i in range(10000, 20000):
        pdf_cifar100_base[i - 10000] = torch.exp(dist_in_base.log_prob(torch.tensor(error_base[i])))
        pdf_cifar100_sn[i - 10000] = torch.exp(dist_in_sn.log_prob(torch.tensor(error_sn[i])))
    for i in range(10000):
        pdf_cifar10_base[i] = torch.exp(dist_in_base.log_prob(torch.tensor(error_base[i])))
        pdf_cifar10_sn[i] = torch.exp(dist_in_sn.log_prob(torch.tensor(error_sn[i])))
    res = calc_auroc(pdf_cifar10_base.numpy(), pdf_cifar100_base.numpy())
    res_sn = calc_auroc(pdf_cifar10_sn.numpy(), pdf_cifar100_sn.numpy())
    print(res_sn)
    print(res)
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
    model = model.load_from_checkpoint(checkpoint_path=os.path.join(opt.model_path, 'best-checkpoint-v6.ckpt'))
    model.cuda()
    with torch.no_grad():
        prob2, prob = [], []
        data_module.setup(stage="test")
        for idx, (images, labels) in enumerate(data_module.test_dataloader()):
            images = images.float().cuda()
            output = model.forward(images)
            res = torch.max(torch.softmax(output, dim=-1), dim=-1).values
            prob.extend(res.cpu().numpy())
        data_module.setup(stage="predict")
        for idx, (images, labels) in enumerate(data_module.predict_dataloader()):
            images = images.float().cuda()
            output = model.forward(images)
            res2 = torch.max(torch.softmax(output, dim=-1), dim=-1).values
            prob2.extend(res2.cpu().numpy())
    print("pass")
    prob = numpy.array(prob)
    prob2 = numpy.array(prob2)
    print(prob)
    print(pdf_cifar10_base)
    f_prob= numpy.maximum(prob, pdf_cifar10_base)
    f_prob2 = numpy.minimum(prob2, pdf_cifar100_base)
    print(calc_auroc(f_prob, f_prob2))


if __name__ == "__main__":
    main()
