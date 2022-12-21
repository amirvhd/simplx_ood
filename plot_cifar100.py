import numpy
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch

from ood_metrics import calc_metrics


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
        pdf_cifar100_base[i-10000] = torch.exp(dist_in_base.log_prob(torch.tensor(error_base[i])))
        pdf_cifar100_sn[i-10000] = torch.exp(dist_in_sn.log_prob(torch.tensor(error_sn[i])))
    for i in range(10000):
        pdf_cifar10_base[i] = torch.exp(dist_in_base.log_prob(torch.tensor(error_base[i])))
        pdf_cifar10_sn[i] = torch.exp(dist_in_sn.log_prob(torch.tensor(error_sn[i])))
    res = calc_auroc(pdf_cifar10_base.numpy(), pdf_cifar100_base.numpy())
    res_sn = calc_auroc(pdf_cifar10_sn.numpy(), pdf_cifar100_sn.numpy())
    print(res_sn)
    print(res)

if __name__ == "__main__":
    main()
