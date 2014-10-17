'''
utils module.
'''

from __future__ import division, print_function

import os

import pandas as pd
import numpy as np
import inspect
from datetime import timedelta

def read_annotations(file_path, columns, sampling_rate):    
    print('Reading annotations...')
    anndf = pd.read_csv('experiments/ecg1_chfdbchf13/annotations.csv',
                        parse_dates=True, index_col='timestamp')
    tmp = []
    for r in anndf.iterrows():
        new_index = r[0]
        new_index += timedelta(microseconds=sampling_rate -
                                            r[0].microsecond%sampling_rate)
        tmp.append([new_index] + list(r[1]))
    anndf = pd.DataFrame(tmp, columns=['timestamp', 'SampleNro', 'Type',
                                       'Sub', 'Chan', 'Num', 'Aux'])
    anndf = anndf.set_index('timestamp')
    anndf[columns] = anndf[columns].astype(np.str)
    return anndf

def _plot_annotations(annotations, df, column, ytext, ax):
    for r in annotations.iteritems():
        x = ax.convert_xunits(r[0])
        y = ax.convert_yunits(df[column][r[0]])
        ax.annotate(r[1], xy=(x, y), xytext=(x, ytext))
        ax.axvline(x, color='r', linewidth=0.75)

def plot_results(df, data_columns, score_column, likelihood_column,
                 match, slce=None, show_plot=True, save_plot=False,
                 cut_percentile=75, axhlines=[0.5, 0.97725, 0.999968]):
    import matplotlib as mpl
    mpl.use('Agg')

    import pylab
    import seaborn as sns

    if slce is not None:
        df = df[slce]
    m_data = np.max(df[data_columns])[0]
    m_score = np.max(df[score_column])
    indexer = df.Annotation.str.match(match, na=False, as_indexer=True)
    annotations = df.Annotation[indexer]
    with sns.color_palette('Set2') as p:
        f = pylab.figure()
        pylab.subplot(3,1,1)
        ax1 = f.gca()
        df[data_columns].plot(ax=ax1, alpha=0.7)
        ax1.set_ylabel(str(data_columns))
        _plot_annotations(annotations, df, data_columns[0], m_data, ax1)
        pylab.legend()
        pylab.subplot(3,1,2)
        ax2 = f.gca()
        df[likelihood_column].plot(ax=ax2, color=p[1], alpha=0.7,
                                   ylim=(0, 1.2))
        ax2.set_ylabel(likelihood_column)
        for hline in axhlines:
            ax2.axhline(hline, color='b', linewidth=0.75)
        _plot_annotations(annotations, df, data_columns[0], 1.1, ax2)
        pylab.legend()
        pylab.subplot(3,1,3)
        ax3 = f.gca()
        scores = df[score_column]
        upper_percentile = np.percentile(scores, cut_percentile)
        scores[scores > upper_percentile] = upper_percentile
        scores.plot(ax=ax3, color=p[2], alpha=0.7)
        ax3.set_ylabel(score_column)
        _plot_annotations(annotations, df, data_columns[0], m_score, ax3)
        pylab.legend()

def f1_score(df, col, thrs, match):
    detected = len(df[df[col] >thrs])
    annotation_indexer = df.Annotation.str.match(match, na=False,
                                                 as_indexer=True)
    col_thrs = df[col] > thrs
    real = len(df[annotation_indexer])
    true_positives = len(df[col_thrs & annotation_indexer])
    false_positives = len(df[col_thrs & ~annotation_indexer])
    false_negatives = len(df[~col_thrs & annotation_indexer])
    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)
    score = 2*(precision*recall)/(precision+recall)
    return {'column': col, 'threshold': thrs, 'detected': detected,
            'real': real, 'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision, 'recall': recall, 'F1': score}
