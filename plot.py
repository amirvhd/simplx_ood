import numpy
import pickle as pkl
import matplotlib.pyplot as plt

def main():
    with open('./experiments/results/cifar/outlier/simplex_cv0.pkl', 'rb') as f:
        data_base = pkl.load(f)
    with open('./experiments/results/cifar/outlier_sn/simplex_cv0.pkl', 'rb') as f:
        data_sn = pkl.load(f)
    pred_base = data_base.latent_approx()
    error_base = ((data_base.test_latent_reps - pred_base) ** 2).sum(1).cpu().numpy()
    pred_sn = data_sn.latent_approx()
    error_sn = ((data_sn.test_latent_reps - pred_sn) ** 2).sum(1).cpu().numpy()
    sorted_error_base = numpy.argsort(error_base)
    sorted_error_base[sorted_error_base <= 10000] = 0
    sorted_error_base[sorted_error_base > 10000] = 1
    sorted_error_sn = numpy.argsort(error_sn)
    sorted_error_sn[sorted_error_sn <= 10000] = 0
    sorted_error_sn[sorted_error_sn > 10000] = 1
    # print(sum(error_base[:10000]))
    # print(sum(error_sn[:10000]))
    # print(sum(error_base[10000:]))
    # print(sum(error_sn[10000:]))
    print(sum(sorted_error_base[:10000]))
    print(sum(sorted_error_sn[:10000]))
    # fig = plt.figure()
    # plt.imshow(data, cmap="gray", interpolation="none")
    # plt.title("CIFAR100-ood-detection")
    # plt.xticks([])
    # plt.yticks([])
if __name__ == "__main__":

    main()
