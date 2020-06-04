import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_dir)

import multiprocessing
import time

import glob
import numpy as np
import scipy.signal as signal
import scipy.io as spio

from configs.dataset import *
from configs.parameters import *


def do_work(item):
    fn, slide_ratio, sample_length_sec, sample_slide_sec, fs, output_dir, selection = item
    time.sleep(.1)  # pretend to do some lengthy work.

    sample_length = int(sample_length_sec * fs)
    sample_slide = int(sample_slide_sec * slide_ratio * fs)

    mat = spio.loadmat(fn)
    data = mat['data']

    n_sample = (data.shape[1] - sample_length) // sample_slide + 1

    xs = []
    ys = []

    identity = os.path.split(fn)[-1].split(".")[0]

    for idx in range(n_sample):
        start = max(idx * sample_slide, 0)
        end = start + sample_length
        sample = data[:, start:end]

        check = np.std(sample, axis=1)

        if np.any(check == 0):
            # drop bad slices
            pass
        else:
            _, x = signal.welch(x=sample, fs=fs, nperseg=fs, noverlap=int(fs * .5))

            if fn.find('Interictal') > 0:
                y = 0
            elif fn.find('Preictal') > 0:
                y = 1
            else:
                raise ValueError('Wrong filename.')
            xs.append(x)
            ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)

    # delta <4
    # theta 4-7
    # alpha 8-15
    # beta 16-31
    # gamma > 32
    # low gamma < 60
    # high gamma >60

    delta = xs[:, :, 1:4]
    theta = xs[:, :, 4:8]
    alpha = xs[:, :, 8:16]
    beta = xs[:, :, 16:32]
    low_gamma = np.concatenate((xs[:, :, 32:47], xs[:, :, 54:61]), axis=2)
    high_gamma = np.concatenate((xs[:, :, 61:97], xs[:, :, 104:]), axis=2)

    bands = []
    bands.append(delta)
    bands.append(theta)
    bands.append(alpha)
    bands.append(beta)
    bands.append(low_gamma)
    bands.append(high_gamma)

    current = []
    for id in selection:
        current.append(bands[id])

    xs = np.concatenate(current, axis=2)

    print("identity {}: data shape {} & label shape {}".format(
        identity, xs.shape, ys.shape))

    filename = os.path.join(output_dir, identity + ".mat")
    spio.savemat(
        file_name=filename,
        mdict={
            'data': xs,
            'label': ys,
        }
    )


if __name__ == "__main__":

    data_dir = os.path.join(raw_data_dir, "MAT_MERGE")

    output_dir = os.path.join(psd_feature_dir)

    if os.path.exists(output_dir):
        pass
    else:
        os.makedirs(output_dir)

    for target in targets:

        inter_fns = glob.glob(os.path.join(data_dir, "{}/{}*.mat".format("Interictal", target)))
        pre_fns = glob.glob(os.path.join(data_dir, "{}/{}*.mat".format("Preictal_ten_minutes", target)))
        inter_statistics_length = 0
        pre_statistics_length = 0

        for fn in pre_fns:
            mat = spio.loadmat(fn)
            data = mat['data']
            pre_statistics_length += data.shape[1] // fs

        for fn in inter_fns:
            mat = spio.loadmat(fn)
            data = mat['data']
            inter_statistics_length += data.shape[1] // fs

        inter_pre_ratio = 1

        print("length (second): preictal {:10d}, interictal {:10d}, ratio {:.5f}".format(
            pre_statistics_length, inter_statistics_length, inter_pre_ratio)
        )

        start = time.perf_counter()

        args = []
        for fn in inter_fns:
            args.append([fn, inter_pre_ratio, feature_length_sec, feature_slide_sec, fs, output_dir, selection])
        for fn in pre_fns:
            args.append([fn, 1, feature_length_sec, feature_slide_sec, fs, output_dir, selection])

        with multiprocessing.Pool(processes=10) as pool:
            pool.map(do_work, args)

        print('elapsed time:', time.perf_counter() - start)
