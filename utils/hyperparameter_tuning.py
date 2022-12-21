import random
from math import log10
from itertools import product
from torch.nn import Sigmoid, Tanh, LeakyReLU, ReLU
from torch.optim import SGD, Adam
from Trainer.cifar10_trainer import classifier_module
ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item']


def grid_search(grid_search_spaces={
    "lr": [0.0001, 0.001, 0.01, 0.1],
    "wd": [1e-4, 1e-5, 1e-6]
}, trainer=None, model=None, data_module=None):
    """
    A simple grid search based on nested loops to tune learning rate and
    regularization strengths.
    Keep in mind that you should not use grid search for higher-dimensional
    parameter tuning, as the search space explodes quickly.
    """
    configs = []


    # More general implementation using itertools
    for instance in product(*grid_search_spaces.values()):
        configs.append(dict(zip(grid_search_spaces.keys(), instance)))

    return findBestConfig(configs, trainer, model, data_module)


def random_search(random_search_spaces={
    "lr": ([0.0001, 0.1], 'log'),
    "wd": ([1e-7, 1e-4], 'log'),
}, num_search=20, trainer=None, model=None, data_module=None):
    """
    Samples N_SEARCH hyper parameter sets within the provided search spaces
    and returns the best model.
    """
    configs = []
    for _ in range(num_search):
        configs.append(random_search_spaces_to_config(random_search_spaces))

    return findBestConfig(configs, trainer, model, data_module)


def findBestConfig(configs, trainer, model, data_module):
    """
    Get a list of hyperparameter configs for random search or grid search,
    trains a model on all configs and returns the one performing best
    on validation set
    """

    best_val = None
    best_config = None
    best_model = None
    results = []

    for i in range(len(configs)):
        print("\nEvaluating Config #{} [of {}]:\n".format(
            (i + 1), len(configs)), configs[i])
        model = classifier_module(**configs[i])
        trainer.fit(model, data_module)
        results.append(trainer.callback_metrics["val_acc"].numpy())

        if not best_val or trainer.callback_metrics["val_acc"] > best_val:
            best_val, best_model, \
            best_config = trainer.callback_metrics["val_acc"], model, configs[i]
    print("\nSearch done. Best Val accuracy = {}".format(best_val))
    print("Best Config:", best_config)
    return best_model, configs, results


def random_search_spaces_to_config(random_search_spaces):
    """"
    Takes search spaces for random search as input; samples accordingly
    from these spaces and returns the sampled hyper-params as a config-object,
    which will be used to construct solver & network
    """

    config = {}

    for key, (rng, mode) in random_search_spaces.items():
        if mode not in ALLOWED_RANDOM_SEARCH_PARAMS:
            print("'{}' is not a valid random sampling mode. "
                  "Ignoring hyper-param '{}'".format(mode, key))
        elif mode == "log":
            if rng[0] <= 0 or rng[-1] <= 0:
                print("Invalid value encountered for logarithmic sampling "
                      "of '{}'. Ignoring this hyper param.".format(key))
                continue
            sample = random.uniform(log10(rng[0]), log10(rng[-1]))
            config[key] = 10 ** (sample)
        elif mode == "int":
            config[key] = random.randint(rng[0], rng[-1])
        elif mode == "float":
            config[key] = random.uniform(rng[0], rng[-1])
        elif mode == "item":
            config[key] = random.choice(rng)

    return config
