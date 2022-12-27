import numpy
import pickle as pkl
import matplotlib.pyplot as plt
import argparse


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--ood_dataset', type=str, default='cifar100',
                        choices=['cifar100', 'svhn'], help='dataset')

    opt = parser.parse_args()
    return opt


def main():
    opt = parse_option()
    if opt.ood_dataset == 'cifar100':
        with open('./experiments/results/cifar/ood/simplex_ood_cifar100.pkl', 'rb') as f:
            data_base = pkl.load(f)
        with open('./experiments/results/cifar/ood_sn/simplex_ood_cifar100.pkl', 'rb') as f:
            data_sn = pkl.load(f)
    if opt.ood_dataset == 'svhn':
        with open('./experiments/results/cifar/ood/simplex_ood_svhn.pkl', 'rb') as f:
            data_base = pkl.load(f)
        with open('./experiments/results/cifar/ood_sn/simplex_ood_svhn.pkl', 'rb') as f:
            data_sn = pkl.load(f)
    pred_base = data_base.latent_approx()
    error_base = ((data_base.test_latent_reps - pred_base) ** 2).sum(1).cpu().numpy()
    pred_sn = data_sn.latent_approx()
    error_sn = ((data_sn.test_latent_reps - pred_sn) ** 2).sum(1).cpu().numpy()

    sorted_error_base = numpy.flip(numpy.argsort(error_base))
    sorted_error_base[sorted_error_base <= 10000] = 0
    sorted_error_base[sorted_error_base > 10000] = 1
    sorted_error_sn = numpy.flip(numpy.argsort(error_sn))
    sorted_error_sn[sorted_error_sn <= 10000] = 0
    sorted_error_sn[sorted_error_sn > 10000] = 1

    cumulative = numpy.cumsum(sorted_error_base)
    cumulative2 = numpy.cumsum(sorted_error_sn)
    # random
    x = numpy.random.randint(2, size=20000)
    cumulative3 = numpy.cumsum(x)
    # Best case
    x_best = numpy.zeros(20000)
    x_best[:10000] = 1
    cumulative4 = numpy.cumsum(x_best)
    plt.plot(cumulative, c='blue', label="Base model")
    plt.plot(cumulative2, c='green', label="Model with spectral normalization")
    plt.plot(cumulative3, c='grey', label="Random")
    plt.plot(cumulative4, c='brown', label="Maximal")
    plt.xlabel("Number of images inspected")
    plt.ylabel(f"Number of {opt.ood_dataset} detected")
    plt.legend(loc="lower right")
    plt.savefig('ood.pdf', format="pdf")


if __name__ == "__main__":
    main()
