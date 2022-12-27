import argparse
import os
import pickle as pkl
from pathlib import Path

import sklearn
import torch

import torchvision
from torch.utils.data import DataLoader, Dataset

from explainers.simplex import Simplex
from models.new_model import WideResNet


# Load data


def load_svhn(
        batch_size: int, split="test", subset_size=None, shuffle: bool = True
) -> DataLoader:
    mean = (0.4376821, 0.4437697, 0.47280442)
    std = (0.19803012, 0.20101562, 0.19703614)
    dataset = torchvision.datasets.SVHN(
        "~/DATA2/",
        split=split,
        download=False,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ]
        ),
    )
    if subset_size:
        dataset = torch.utils.data.Subset(
            dataset, torch.randperm(len(dataset))[:subset_size]
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_cifar10(
        batch_size: int, train: bool, subset_size=None, shuffle: bool = True
) -> DataLoader:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    dataset = torchvision.datasets.CIFAR10(
        "~/DATA2/",
        train=train,
        download=False,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ]
        ),
    )
    if subset_size:
        dataset = torch.utils.data.Subset(
            dataset, torch.randperm(len(dataset))[:subset_size]
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_cifar100(
        batch_size: int, train: bool, subset_size=None, shuffle: bool = True
) -> DataLoader:
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    dataset = torchvision.datasets.CIFAR100(
        "~/DATA2/",
        train,
        download=False,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ]
        ),
    )
    if subset_size:
        dataset = torch.utils.data.Subset(
            dataset, torch.randperm(len(dataset))[:subset_size]
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def approximation_quality(
        random_seed: int = 42,
) -> None:
    print(
        100 * "-"
        + "\n"
        + "Welcome in the approximation quality experiment for CIFAR-10. \n"
          f"Settings: random_seed = {random_seed} "
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model

    classifier1 = WideResNet(spectral_conv=False, spectral_bn=False)
    new_state_dict = {}
    state_dict = torch.load(
        os.path.join("/dss/dssmcmlfs01/pn69za/pn69za-dss-0002/ra49bid2/saved_models/BERD/", "best-checkpoint-v1.ckpt"),
        map_location=torch.device('cpu'))[
        "state_dict"]
    for k, v in state_dict.items():
        k = k.replace("model.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    classifier1.load_state_dict(state_dict)
    classifier1.to(device)
    classifier1.eval()

    new_state_dict = {}
    state_dict = torch.load(
        os.path.join("/dss/dssmcmlfs01/pn69za/pn69za-dss-0002/ra49bid2/saved_models/BERD/", "best-checkpoint-v2.ckpt"),
        map_location=torch.device('cpu'))[
        "state_dict"]
    for k, v in state_dict.items():
        k = k.replace("model.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    classifier2 = WideResNet(spectral_conv=True, spectral_bn=True)
    classifier2.load_state_dict(state_dict)
    classifier2.to(device)
    classifier2.eval()

    with open('./experiments/results/cifar/outlier/simplex.pkl', 'rb') as f:
        data_base = pkl.load(f)
    with open('./experiments/results/cifar/outlier_sn/simplex.pkl', 'rb') as f:
        data_sn = pkl.load(f)

    # Load the explainer
    latent_rep_approx = data_base.latent_approx()[:10000]
    latent_rep_true = data_base.test_latent_reps[:10000]
    output_approx = classifier1.latent_to_presoftmax(latent_rep_approx).detach()
    output_true = classifier1.latent_to_presoftmax(latent_rep_true).detach()
    latent_r2_score = sklearn.metrics.r2_score(
        latent_rep_true.cpu().numpy(), latent_rep_approx.cpu().numpy()
    )
    output_r2_score = sklearn.metrics.r2_score(
        output_true.cpu().numpy(), output_approx.cpu().numpy()
    )
    print(
        f"base latent r2: {latent_r2_score:.2g} ; base output r2 = {output_r2_score:.2g}."
    )
    latent_rep_approx = data_sn.latent_approx()[:10000]
    latent_rep_true = data_sn.test_latent_reps[:10000]
    output_approx = classifier2.latent_to_presoftmax(latent_rep_approx).detach()
    output_true = classifier2.latent_to_presoftmax(latent_rep_true).detach()
    latent_r2_score = sklearn.metrics.r2_score(
        latent_rep_true.cpu().numpy(), latent_rep_approx.cpu().numpy()
    )
    output_r2_score = sklearn.metrics.r2_score(
        output_true.cpu().numpy(), output_approx.cpu().numpy()
    )
    print(
        f"SN latent r2: {latent_r2_score:.2g} ; base output r2 = {output_r2_score:.2g}."
    )


def ood_detection(
        ood: str = "CIFAR100",
        random_seed: int = 42,
        save_path: str = "experiments/results/cifar/ood/",
        train: bool = False,
) -> None:
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epoch_simplex = 10000

    print(
        100 * "-" + "\n" + "Welcome in the outlier detection experiment for CIFAR10. \n"
                           f"Settings: random_seed = {random_seed} ; ood = {ood}.\n"
    )
    current_path = Path.cwd()
    print(current_path)
    save_path = current_path / save_path
    print(save_path)
    # Create saving directory if inexistent
    if not save_path.exists():
        print("pass")
        print(f"Creating the saving directory {save_path}")
        os.makedirs(save_path)

    # Load the model
    # Classifier1 is wide-resnet28-10
    save_path1 = current_path / "experiments/results/cifar/ood/"
    classifier1 = WideResNet(spectral_conv=False, spectral_bn=False)
    new_state_dict = {}
    state_dict = torch.load(
        os.path.join(save_path1, "best-checkpoint-v1.ckpt"),
        map_location=torch.device('cpu'))[
        "state_dict"]
    for k, v in state_dict.items():
        k = k.replace("model.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    classifier1.load_state_dict(state_dict)
    classifier1.to(device)
    classifier1.eval()
    # Classifier2 is wide-resnet28-10 with  spectral normalization
    save_path2 = current_path / "experiments/results/cifar/ood_sn/"
    new_state_dict = {}
    state_dict = torch.load(
        os.path.join(save_path2, "best-checkpoint-v2.ckpt"),
        map_location=torch.device('cpu'))[
        "state_dict"]
    for k, v in state_dict.items():
        k = k.replace("model.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    classifier2 = WideResNet(spectral_conv=True, spectral_bn=True)
    classifier2.load_state_dict(state_dict)
    classifier2.to(device)
    classifier2.eval()

    # Load data:
    corpus_loader = load_cifar10(batch_size=1000, train=True)
    cifar10_test_loader = load_cifar10(batch_size=1000, train=False)

    corpus_latent_reps1 = []
    corpus_latent_reps2 = []
    corpus_features = []
    for i, (corpus_feature, _) in enumerate(corpus_loader):
        corpus_features.append(corpus_feature)
        corpus_latent_reps1.append(classifier1.latent_representation(corpus_feature.to(device).detach()).detach())
        corpus_latent_reps2.append(classifier2.latent_representation(corpus_feature.to(device).detach()).detach())
    corpus_features = torch.cat(corpus_features, dim=0).to(device).detach()
    corpus_latent_reps1 = torch.cat(corpus_latent_reps1, dim=0).to(device).detach()
    corpus_latent_reps2 = torch.cat(corpus_latent_reps2, dim=0).to(device).detach()

    cifar10_test_features = []
    cifar10_test_latent_reps1 = []
    cifar10_test_latent_reps2 = []
    for i, (cifar10_test_feature, _) in enumerate(cifar10_test_loader):
        cifar10_test_features.append(cifar10_test_feature)
        cifar10_test_latent_reps1.append(
            classifier1.latent_representation(cifar10_test_feature.to(device).detach()).detach())
        cifar10_test_latent_reps2.append(
            classifier2.latent_representation(cifar10_test_feature.to(device).detach()).detach())
    cifar10_test_features = torch.cat(cifar10_test_features, dim=0).to(device).detach()
    cifar10_test_latent_reps1 = torch.cat(cifar10_test_latent_reps1, dim=0).to(device).detach()
    cifar10_test_latent_reps2 = torch.cat(cifar10_test_latent_reps2, dim=0).to(device).detach()
    if ood == "CIFAR100":
        ood_test_loader = load_cifar100(batch_size=1000, train=False)
    elif ood == "SVHN":
        ood_test_loader = load_svhn(batch_size=1000, split="test")
    else:
        print("Dataset name is not correct")
    ood_test_features = []
    ood_test_latent_reps1 = []
    ood_test_latent_reps2 = []
    for i, (ood_test_feature, _) in enumerate(ood_test_loader):
        ood_test_features.append(ood_test_feature)
        ood_test_latent_reps1.append(
            classifier1.latent_representation(ood_test_feature.to(device).detach()).detach())
        ood_test_latent_reps2.append(
            classifier2.latent_representation(ood_test_feature.to(device).detach()).detach())
    ood_test_features = torch.cat(ood_test_features, dim=0).to(device).detach()
    ood_test_latent_reps1 = torch.cat(ood_test_latent_reps1, dim=0).to(device).detach()
    ood_test_latent_reps2 = torch.cat(ood_test_latent_reps2, dim=0).to(device).detach()
    test_latent_reps1 = torch.cat([cifar10_test_latent_reps1, ood_test_latent_reps1], dim=0)
    test_latent_reps2 = torch.cat([cifar10_test_latent_reps2, ood_test_latent_reps2], dim=0)
    test_features = torch.cat([cifar10_test_features, ood_test_features], dim=0)

    # Fit corpus:
    # SimplEx1 is  for SimplEx fit to the normal wide-resnet
    simplex1 = Simplex(
        corpus_examples=corpus_features, corpus_latent_reps=corpus_latent_reps1
    )
    simplex1.fit(
        test_examples=test_features,
        test_latent_reps=test_latent_reps1,
        n_epoch=n_epoch_simplex,
        reg_factor=0,
        n_keep=corpus_features.shape[0],
    )
    explainer_path = save_path1 / f"simplex_ood{ood}.pkl"
    with open(explainer_path, "wb") as f:
        print(f"Saving simplex decomposition in {explainer_path}.")
        pkl.dump(simplex1, f)
    # SimplEx2 is  for SimplEx fit to the distance-aware wide-resnet
    simplex2 = Simplex(
        corpus_examples=corpus_features, corpus_latent_reps=corpus_latent_reps2
    )
    simplex2.fit(
        test_examples=test_features,
        test_latent_reps=test_latent_reps2,
        n_epoch=n_epoch_simplex,
        reg_factor=0,
        n_keep=corpus_features.shape[0],
    )
    explainer_path = save_path2 / f"simplex_ood{ood}.pkl"
    with open(explainer_path, "wb") as f:
        print(f"Saving simplex decomposition in {explainer_path}.")
        pkl.dump(simplex2, f)


def main(experiment: str, ood_dataset: str) -> None:
    if experiment == "approximation_quality":
        approximation_quality()
    elif experiment == "ood_detection":
        ood_detection(ood=ood_dataset)
    else:
        raise ValueError(
            "The name of the experiment is not valid. "
            "Valid names are: "
            "approximation_quality , ood_detection."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-experiment",
        type=str,
        default="approximation_quality",
        help="Experiment to perform",
    )
    parser.add_argument('--ood_dataset', type=str, default='cifar100',
                        choices=['cifar100', 'svhn'], help='dataset')
    args = parser.parse_args()
    main(args.experiment, args.ood_dataset)
