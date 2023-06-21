import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from sklearn import metrics
from src.transformer_utils.language_loader import wikitext2_loader, ptb_loader


def get_dataset(dataset_name, train_flag, datadir, exp_dict, device):
    if dataset_name == "mnist":
        class MNISTindex(Dataset):
            def __init__(self):
                self.dataset = torchvision.datasets.MNIST(datadir, train=train_flag,
                                       download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.5,), (0.5,))
                                       ]))

            def __getitem__(self, index):
                data, target = self.dataset.__getitem__(index)
                return data, target, index

            def __len__(self):
                return len(self.dataset)

        dataset = MNISTindex()

    if dataset_name == "cifar10":
        class CIFAR10index(Dataset):
            def __init__(self):
                transform_function = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ])

                self.dataset = torchvision.datasets.CIFAR10(
                    root=datadir,
                    train=train_flag,
                    download=True,
                    transform=transform_function)

            def __getitem__(self, index):
                data, target = self.dataset.__getitem__(index)
                return data, target, index

            def __len__(self):
                return len(self.dataset)

        dataset = CIFAR10index()

    if dataset_name == "cifar100":
        class CIFAR100index(Dataset):
            def __init__(self):
                transform_function = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ])

                self.dataset = torchvision.datasets.CIFAR100(
                    root=datadir,
                    train=train_flag,
                    download=True,
                    transform=transform_function)

            def __getitem__(self, index):
                data, target = self.dataset.__getitem__(index)
                return data, target, index

            def __len__(self):
                return len(self.dataset)

        dataset = CIFAR100index()


    if dataset_name == "svhn":
        class SVHNindex(Dataset):
            def __init__(self):
                normalize = transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
                transform = transforms.Compose([transforms.ToTensor(), normalize,])
                if train_flag:
                    split = 'train'
                else:
                    split = 'test'
                self.dataset = torchvision.datasets.SVHN(root=datadir, split=split, download=True, transform=transform)

            def __getitem__(self, index):
                data, target = self.dataset.__getitem__(index)
                return data, target, index

            def __len__(self):
                return len(self.dataset)

        dataset = SVHNindex()

    if dataset_name == "fashion":
        class FashionMNISTindex(Dataset):
            def __init__(self):
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5), (0.5))
                                                ])

                # Download and load the training data
                self.dataset = torchvision.datasets.FashionMNIST(datadir, download=True, train=train_flag, transform=transform)

            def __getitem__(self, index):
                data, target = self.dataset.__getitem__(index)
                return data, target, index

            def __len__(self):
                return len(self.dataset)

        dataset = FashionMNISTindex()

    if dataset_name in ["mushrooms", "w8a",
                        "rcv1", "ijcnn"]:
        sigma_dict = {"mushrooms": 0.5,
                      "w8a":20.0,
                      "rcv1":0.25 ,
                      "ijcnn":0.05}

        X, y = load_libsvm(dataset_name, data_dir=datadir)

        labels = np.unique(y)
        y[y==labels[0]] = 0
        y[y==labels[1]] = 1
        # splits used in experiments
        splits = train_test_split(X, y, test_size=0.2, shuffle=True,
                    random_state=9513451)
        X_train, X_test, Y_train, Y_test = splits

        if train_flag:
            fname_rbf = "%s/rbf_%s_%s_train.npy" % (datadir, dataset_name, sigma_dict[dataset_name])
            if os.path.exists(fname_rbf):
                k_train_X = np.load(fname_rbf)
            else:
                k_train_X = rbf_kernel(X_train, X_train, sigma_dict[dataset_name])
                np.save(fname_rbf, k_train_X)
                print('%s saved' % fname_rbf)

            X_train = k_train_X
            X_train = torch.FloatTensor(X_train)
            Y_train = torch.LongTensor(Y_train)
            indexes = torch.FloatTensor(np.arange(len(Y_train)))

            dataset = torch.utils.data.TensorDataset(X_train, Y_train, indexes)

        else:
            fname_rbf = "%s/rbf_%s_%s_test.npy" % (datadir, dataset_name, sigma_dict[dataset_name])
            if os.path.exists(fname_rbf):
                k_test_X = np.load(fname_rbf)
            else:
                k_test_X = rbf_kernel(X_test, X_train, sigma_dict[dataset_name])
                np.save(fname_rbf, k_test_X)
                print('%s saved' % fname_rbf)

            X_test = k_test_X
            X_test = torch.FloatTensor(X_test)
            Y_test = torch.LongTensor(Y_test)
            indexes = torch.FloatTensor(np.arange(len(Y_test)))

            dataset = torch.utils.data.TensorDataset(X_test, Y_test, indexes)

    if dataset_name == "wikitext" or dataset_name == "wikitext2":
        if "tgt_len" in exp_dict["model_args"]:
            tgt_len = exp_dict["model_args"]["tgt_len"]
        else:
            tgt_len = 150
        return wikitext2_loader(
            int(exp_dict["batch_size"]),
            datadir,
            device,
            tgt_len,
            drop_last=exp_dict.get("drop_last") or False,
            full_batch=exp_dict.get("full_batch") or False,
            shuffle=exp_dict.get("shuffle") or False
        )

    if dataset_name == "ptb":
        if "tgt_len" in exp_dict["model_args"]:
            tgt_len = exp_dict["model_args"]["tgt_len"]
        else:
            tgt_len = 128
        return ptb_loader(
            int(exp_dict["batch_size"]),
            datadir,
            device,
            tgt_len,
            drop_last=exp_dict.get("drop_last") or False,
            full_batch=exp_dict.get("full_batch") or False,
            shuffle=exp_dict.get("shuffle") or False
        )

    return dataset


# ===========================================================
# Helpers
import os
import urllib

import numpy as np
from sklearn.datasets import load_svmlight_file


LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
LIBSVM_DOWNLOAD_FN = {"rcv1"       : "rcv1_train.binary.bz2",
                      "mushrooms"  : "mushrooms",
                      "ijcnn"      : "ijcnn1.tr.bz2",
                      "w8a"        : "w8a"}


def load_libsvm(name, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    fn = LIBSVM_DOWNLOAD_FN[name]
    data_path = os.path.join(data_dir, fn)

    if not os.path.exists(data_path):
        url = urllib.parse.urljoin(LIBSVM_URL, fn)
        print("Downloading from %s" % url)
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")

    X, y = load_svmlight_file(data_path)
    return X, y


def rbf_kernel(A, B, sigma):
    distsq = np.square(metrics.pairwise.pairwise_distances(A, B, metric="euclidean"))
    K = np.exp(-1 * distsq/(2*sigma**2))
    return K
