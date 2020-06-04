import h5py
import numpy as np
import glob
import sklearn.model_selection as model_selection
import multiprocessing
import scipy.io as spio


def split_fns(data_dir, target, seed, postfix=".mat"):
    pre_fns = glob.glob("{}/{}*Pre*{}".format(data_dir, target, postfix))
    inter_fns = glob.glob("{}/{}*Inter*{}".format(data_dir, target, postfix))

    pre_fns = sorted(pre_fns)
    inter_fns = sorted(inter_fns)

    fns = []
    fns.extend(pre_fns)
    fns.extend(inter_fns)
    labels = np.concatenate((np.ones([len(pre_fns)]), np.zeros([len(inter_fns)])), axis=0)
    skf = model_selection.StratifiedKFold(n_splits=len(pre_fns), shuffle=True, random_state=seed)

    tr_idx = []
    te_idx = []

    for tr_idx_fold, te_idx_fold in skf.split(fns, labels):
        tr_idx.append(tr_idx_fold)
        te_idx.append(te_idx_fold)

    tr_fns = []
    te_fns = []
    for idx in range(len(pre_fns)):
        tr_idx_fold, te_idx_fold = tr_idx[idx], te_idx[idx]

        tr_fns_fold = []
        te_fns_fold = []

        [tr_fns_fold.append(fns[idx]) for idx in tr_idx_fold]
        [te_fns_fold.append(fns[idx]) for idx in te_idx_fold]

        tr_fns.append(tr_fns_fold)
        te_fns.append(te_fns_fold)

    return tr_fns, te_fns


def get_data_from_fn(fn):
    mat = spio.loadmat(fn)
    x = mat['data']
    y = np.squeeze(mat['label'])

    return x, y


def get_samples(fns, preprocess=None):
    X, Y = [], []

    with multiprocessing.Pool(processes=10) as pool:
        items = pool.map(get_data_from_fn, fns)
    for item in items:
        x, y = item
        X.extend(x)
        Y.extend(y)
    X = np.array(X)
    Y = np.array(Y)

    if preprocess is None:
        pass
    else:
        X = preprocess(X + 1e-6)

    return X, Y
