#!/usr/bin/env python
import os
import pprint

from nupic.swarming import permutations_runner


def write_model_params(cwd, model_params, model_name=None):
    if model_name is None:
        out_dir = os.path.join(cwd, 'model_params')
    else:
        out_dir = os.path.join(cwd, '{}_model_params'.format(model_name))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    model_params_init = os.path.join(out_dir, '__init__.py')
    with open(model_params_init, 'w') as f:
        f.write('\n')
    
    out_path = os.path.join(out_dir, 'model_params.py')
    pp = pprint.PrettyPrinter(indent=4)
    model_params_str = pp.pformat(model_params)
    with open(out_path, 'wb') as out_file:
        out_file.write('MODEL_PARAMS = (\n{}\n)'.format(model_params_str))


def swarm(cwd, input_file, swarm_description, model_name=None,
          write_model=False):
    swarm_work_dir = os.path.abspath('swarm')
    if not os.path.exists(swarm_work_dir):
        os.mkdir(swarm_work_dir)
    stream = swarm_description['streamDef']['streams'][0]
    full_path = os.path.join(cwd, input_file)
    stream['source'] = 'file://{}'.format(full_path)
    label = swarm_description['streamDef']['info']
    model_params = permutations_runner.runWithConfig(
                                      swarm_description,
                                      {'maxWorkers': 4, 'overwrite': True},
                                      outputLabel=label,
                                      outDir=swarm_work_dir,
                                      permWorkDir=swarm_work_dir
                                                    )
    if write_model:
        write_model_params(cwd, model_params, model_name)
    return model_params


if __name__ == '__main__':
    from swarm_description import SWARM_DESCRIPTION
    import sys
    args = sys.argv
    if '--input_file' in args:
        input_file = args[args.index('--input_file') + 1]
    else:
        input_file = 'data.csv'
    if '--cwd' in args:
        cwd = args[args.index('--cwd') + 1]
    else:
        cwd = os.getcwd()
    swarm(cwd, input_file, SWARM_DESCRIPTION)
