import Oger as og
import numpy as np
import pandas as pd


def _modify_signal(signal, n_samples, sample_len, increases,
                   start_increase, stop_increase,
                   start_decrease, stop_decrease,
                   in_place=False):
    increase_len = stop_increase - start_increase
    increases = np.asarray(increases, dtype=float)
    range_increase = increases * np.mgrid[:increase_len,:n_samples][0]/increase_len
    if in_place:
        ret_val = signal
    else:
        ret_val = signal.copy()
    ret_val[start_increase:stop_increase] += range_increase

    ret_val[stop_increase:start_decrease] += increases

    increase_len = stop_decrease - start_decrease
    range_increase = increases * np.mgrid[increase_len:0:-1,:n_samples][0]/increase_len
    ret_val[start_decrease:stop_decrease] += range_increase
    
    columns = []
    for i in range(n_samples):
        columns.append('signal_{}_samples_{}d'.format(sample_len, i))
    ret_val = pd.DataFrame(data=ret_val, columns=columns,
                           index=pd.Index(np.arange(sample_len),
                                          name='step'))
    annotations = pd.DataFrame(data=np.array(['']*sample_len, dtype=np.str),
                               columns=['Annotation'],
                               index=pd.Index(np.arange(sample_len),
                                              name='step'))
    annotations['Annotation'][start_increase:stop_decrease] = 'A'    

    return ret_val, annotations


def generate_narma30(start_increase, stop_increase,
                     start_decrease, stop_decrease,
                     n_samples=1, sample_len=1000, increases=[1],
                     csv_filename=None, annotations_filename=None):
    signal = og.datasets.narma30(n_samples=n_samples, sample_len=sample_len)
    signal = np.asarray(signal[0]).reshape(n_samples, sample_len).T

    signal, annotations = _modify_signal(signal, n_samples, sample_len,
                                         increases, start_increase,
                                         stop_increase, start_decrease,
                                         stop_decrease, in_place=True)

    if csv_filename is not None:
        signal.to_csv(csv_filename)

    if annotations_filename is not None:
        signal.to_csv(annotations_filename)

    return signal, annotations


def generate_mackey_glass(start_increase, stop_increase,
                          start_decrease, stop_decrease,
                          n_samples=1, sample_len=1000, increases=[1],
                          csv_filename=None, annotations_filename=None,
                          seed=None):
    signal = og.datasets.mackey_glass(n_samples=n_samples, seed=seed,
                                      sample_len=sample_len)
    signal = np.asarray(signal).reshape(n_samples, sample_len).T

    signal, annotations = _modify_signal(signal, n_samples, sample_len,
                                         increases, start_increase,
                                         stop_increase, start_decrease,
                                         stop_decrease, in_place=True)

    if csv_filename is not None:
        signal.to_csv(csv_filename)

    if annotations_filename is not None:
        signal.to_csv(annotations_filename)

    return signal, annotations


def generate_memtest(start_increase, stop_increase,
                     start_decrease, stop_decrease,
                     n_samples=1, sample_len=1000, increases=[1],
                     csv_filename=None, annotations_filename=None):
    signal = og.datasets.memtest(n_samples=n_samples, sample_len=sample_len)
    signal = np.asarray(signal[0]).reshape(n_samples, sample_len).T

    signal, annotations = _modify_signal(signal, n_samples, sample_len,
                                         increases, start_increase,
                                         stop_increase, start_decrease,
                                         stop_decrease, in_place=True)

    if csv_filename is not None:
        signal.to_csv(csv_filename)

    if annotations_filename is not None:
        signal.to_csv(annotations_filename)

    return signal, annotations


def generate_lorentz(start_increase, stop_increase,
                     start_decrease, stop_decrease, sample_len=1000,
                     increases=[1, 1, 1], csv_filename=None,
                     annotations_filename=None):
    signal = og.datasets.lorentz(sample_len=sample_len)
    signal = np.asarray(signal[0]).reshape(3, sample_len).T

    signal, annotations = _modify_signal(signal, 3, sample_len,
                                         increases, start_increase,
                                         stop_increase, start_decrease,
                                         stop_decrease, in_place=True)

    if csv_filename is not None:
        signal.to_csv(csv_filename)

    if annotations_filename is not None:
        signal.to_csv(annotations_filename)

    return signal, annotations


