import numpy
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def main():
    # with open('./experiments/results/cifar/outlier/simplex_cv0.pkl', 'rb') as f:
    #     data_base = pkl.load(f)
    # with open('./experiments/results/cifar/outlier_sn/simplex_cv0.pkl', 'rb') as f:
    #     data_sn = pkl.load(f)
    # pred_base = data_base.latent_approx()
    # error_base = ((data_base.test_latent_reps - pred_base) ** 2).sum(1).cpu().numpy()
    # pred_sn = data_sn.latent_approx()
    # error_sn = ((data_sn.test_latent_reps - pred_sn) ** 2).sum(1).cpu().numpy()
    with open('./experiments/results/cifar/outlier/simplex_svhn1_cv0.pkl', 'rb') as f:
        data_base1 = pkl.load(f)
    with open('./experiments/results/cifar/outlier_sn/simplex_svhn1_cv0.pkl', 'rb') as f:
        data_sn1 = pkl.load(f)
    with open('./experiments/results/cifar/outlier/simplex_svhn2_cv0.pkl', 'rb') as f:
        data_base2 = pkl.load(f)
    with open('./experiments/results/cifar/outlier_sn/simplex_svhn2_cv0.pkl', 'rb') as f:
        data_sn2 = pkl.load(f)
    pred_base1 = data_base1.latent_approx()
    error_base1 = ((data_base1.test_latent_reps - pred_base1) ** 2).sum(1).cpu().numpy()
    pred_sn1 = data_sn1.latent_approx()
    error_sn1 = ((data_sn1.test_latent_reps - pred_sn1) ** 2).sum(1).cpu().numpy()
    pred_base2 = data_base2.latent_approx()
    error_base2 = ((data_base2.test_latent_reps - pred_base2) ** 2).sum(1).cpu().numpy()
    pred_sn2 = data_sn2.latent_approx()
    error_sn2 = ((data_sn2.test_latent_reps - pred_sn2) ** 2).sum(1).cpu().numpy()
    print(error_base1.shape)
    error_base = numpy.concatenate(error_base1, error_base2)
    error_sn = numpy.concatenate(error_sn1, error_sn2)
    sorted_error_base = numpy.flip(numpy.argsort(error_base))
    sorted_error_base[sorted_error_base <= 10000] = 0
    sorted_error_base[sorted_error_base > 10000] = 1
    sorted_error_sn = numpy.flip(numpy.argsort(error_sn))
    sorted_error_sn[sorted_error_sn <= 10000] = 0
    sorted_error_sn[sorted_error_sn > 10000] = 1
    # print(sum(error_base[:10000]))
    # print(sum(error_sn[:10000]))
    # print(sum(error_base[10000:]))
    # print(sum(error_sn[10000:]))
    # print(sorted_error_base[:10])
    # print(sum(sorted_error_sn[:10000]))
    cumulative = numpy.cumsum(sorted_error_base)
    cumulative2 = numpy.cumsum(sorted_error_sn)
    x = numpy.random.randint(2, size=20000)
    cumulative3 = numpy.cumsum(x)
    x_best = numpy.zeros(20000)
    x_best[:10000] = 1
    cumulative4 = numpy.cumsum(x_best)
    print(roc_auc_score(x_best, sorted_error_base))
    print(roc_auc_score(x_best, sorted_error_sn))
    plt.plot(cumulative, c='blue', label="Base model")
    plt.plot(cumulative2, c='green', label="Model with spectral normalization")
    plt.plot(cumulative3, c='grey', label="Random")
    plt.plot(cumulative4, c='brown', label="Maximal")
    plt.title("SVHN100-ood-detection")
    plt.xlabel("Number of images inspected")
    plt.ylabel("Number of SVHN detected")
    plt.legend(loc="lower right")
    plt.savefig('SVHN_cifar10.png')
    # fig = plt.figure()
    # plt.imshow(data, cmap="gray", interpolation="none")
    #


if __name__ == "__main__":
    main()
