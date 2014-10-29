#!/usr/bin/env python
# ----------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
# ----------------------------------------------------------------------
'''
Experiment on respiratory data stream using
the motion-artifact/eeg_1 dataset from PhysioNet.
@author: Mario Tambos
'''
from __future__ import division, print_function

import os

import pandas as pd
import numpy as np
import inspect
from datetime import timedelta, datetime
import shutil

import utils


def set_annotations_and_plot(file_name, anndf, likelohood_column, plot):
    print('Reading results...')
    df = pd.read_csv(file_name, parse_dates=True, index_col='timestamp')
    df['Annotation'] = anndf.Annotation
    print('Writing annotated results...')
    df.to_csv(file_name)
    if plot:
        utils.plot_results(df, 'ACC_z1', 'anomaly_score',
                           likelohood_column, '[A-F]')
    return df


def fill_swarm_description(file_path, buffer_len, predicted_field):
    swarm_desc = {
        'includedFields': [
            {
                'fieldName': 'timestamp',
                'fieldType': 'datetime',
            },
            {
                'fieldName': 'ACC_z1',
                'fieldType': 'float',
            },
        ],
        'streamDef': {
            'info': 'motion-artifact_eeg_1',
            'version': 1,
            'streams': [
                {
                    'info': 'motion-artifact_eeg_1',
                    'source': file_path,
                    'columns': ['*']
                }
            ]
        },
        'inferenceType': 'TemporalAnomaly',
        'inferenceArgs': {
            'predictionSteps': [1],
            'predictedField': predicted_field
        },
        'iterationCount': buffer_len,
        'swarmSize': 'large'
    }
    return swarm_desc


def main(cwd, do_amgng, amgng_file, ma_window, ma_recalc_delay,
         do_cla, cla_file, buffer_len, plot):
    values = inspect.getargvalues(inspect.currentframe())[3]
    print('using parameters: {}'.format(values))
    annotations_path = os.path.join(cwd, 'annotations.csv')
    anndf = utils.read_annotations(annotations_path, ['Annotation'])

    amgng_df = None
    if do_amgng:
        from mgng.amgng import main as amgng_main
        print('Training AMGNG model...')
        out_file = os.path.join(cwd, 'out_amgng_{}'.format(amgng_file))
        full_path = os.path.join(cwd, amgng_file)
        start = datetime.now()
        amgng_main(input_file=full_path, output_file=out_file,
                   buffer_len=buffer_len, index_col='timestamp',
                   skip_rows=[1,2], ma_window=ma_window,
                   ma_recalc_delay=ma_recalc_delay)
        amgng_time = datetime.now() - start
        amgng_df = set_annotations_and_plot(out_file, anndf,
                                            'anomaly_density', plot)
        amgng_df['AnnotationSpans'] = amgng_df.Annotation.copy()
        utils.fill_annotations(amgng_df, 'AnnotationSpans', '[A-F]')
        amgng_df.to_csv(out_file)
        print('Time taken: amgng={}'.format(amgng_time))

    cla_df = None
    if do_cla:
        from cla.swarm import swarm
        from cla.cla import create_model, open_input_file
        from cla.cla import  prepare_run, process_row
        print('Training CLA model...')
        full_path = os.path.join(cwd, cla_file)
        out_file = os.path.join(cwd, 'out_cla_{}'.format(cla_file))
        cla_model = {}
        fields, csv_reader, input_handler = open_input_file(full_path)
        start = datetime.now()
        for p in fields:
            swarm_desc = fill_swarm_description(full_path, buffer_len, p)
            model_params = swarm(cwd=cwd, input_file=cla_file,
                                 swarm_description=swarm_desc)
            model = create_model(params=model_params, predictedField=p)
            model_out_file = os.path.join(cwd, '{}_{}'.format(p, cla_file))
            shifter, output_handler = prepare_run(fields=fields,
                                                  predicted_field=p,
                                                  plot=False,
                                                  output_name=model_out_file)
            cla_model[p] = {'model': model, 'shifter': shifter,
                            'output_handler': output_handler,
                            'model_out_file': model_out_file}

        swarm_time = datetime.now() - start
        start = datetime.now()
        for i, row in enumerate(csv_reader):
            for p in fields:
                process_row(row=row, fields=fields, predicted_field=p,
                            model=cla_model[p]['model'],
                            shifter=cla_model[p]['shifter'],
                            output_handler=cla_model[p]['output_handler'],
                            counter=i)
        cla_time = datetime.now() - start

        input_handler.close()
        for i, p in enumerate(fields):
            cla_model[p]['output_handler'].close()
            df = pd.read_csv(cla_model[p]['model_out_file'], parse_dates=True,
                             index_col='timestamp')
            if i == 0:
                cla_df = df
            else:
                cla_df.anomaly_likelihood += df.anomaly_likelihood
        cla_df.anomaly_likelihood /= len(fields)
        cla_df.to_csv(out_file)
        cla_df = set_annotations_and_plot(out_file, anndf,
                                          'anomaly_likelihood', plot)
        cla_df['AnnotationSpans'] = cla_df.Annotation.copy()
        utils.fill_annotations(cla_df, 'AnnotationSpans', '[A-F]')
        cla_df.to_csv(out_file)
        print('Time taken: swarm={}, cla={}'.format(swarm_time, cla_time))

    return amgng_df, cla_df


if __name__ == '__main__':
    import sys
    args = sys.argv
    if '--do_amgng' in args:
        do_amgng = True
    else:
        do_amgng = False
    if '--amgng_file' in args:
        amgng_file = args[args.index('--amgng_file') + 1]
    else:
        amgng_file = 'experiments/motion-artifact/acc_z1.csv'
    if '--do_cla' in args:
        do_cla = True
    else:
        do_cla = False
    if '--cla_file' in args:
        cla_file = args[args.index('--cla_file') + 1]
    else:
        cla_file = 'experiments/motion-artifact/acc_z1.csv'
    if '--cwd' in args:
        cwd = args[args.index('--cwd') + 1]
    else:
        if do_amgng:
            cwd = os.path.dirname(amgng_file)
            amgng_file = os.path.basename(amgng_file)
        elif do_cla:
            cwd = os.path.dirname(cla_file)
            cla_file = os.path.basename(cla_file)
        else:
            cwd = os.getcwd()
    if '--buffer_len' in args:
        buffer_len = int(args[args.index('--buffer_len') + 1])
    else:
        buffer_len = 7000
    if '--ma_window' in args:
        ma_window = int(args[args.index('--ma_window') + 1])
    else:
        ma_window = 2000
    if '--ma_recalc_delay' in args:
        ma_recalc_delay = int(args[args.index('--ma_recalc_delay') + 1])
    else:
        ma_recalc_delay = 1
    if '--plot' in args:
        plot = True
    else:
        plot = False
    main(cwd=cwd, do_amgng=do_amgng, amgng_file=amgng_file,
         ma_window=ma_window, do_cla=do_cla, cla_file=cla_file,
         buffer_len=buffer_len, plot=plot, ma_recalc_delay=ma_recalc_delay)
