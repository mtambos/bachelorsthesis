#!/usr/bin/env python

import dateutil.parser as du_parser
import csv

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory
import nupic_anomaly_output as nupic_output

def create_model(params, predictedField):
    model = ModelFactory.create(params)
    model.enableInference({'predictedField': predictedField})
    return model


def run_model(model, input_file, plot_name, plot, predicted_field):
    input_file = open(input_file, 'rb')
    csv_reader = csv.reader(input_file)
    # skip header rows
    csv_reader.next()
    csv_reader.next()
    csv_reader.next()

    if plot:
        output = nupic_output.NuPICPlotOutput(y_label=predicted_field, name=plot_name)
    else:
        output = nupic_output.NuPICFileOutput(columns=[predicted_field, 'prediction'],
                                              name=plot_name)
    shifter = InferenceShifter()

    counter = 0
    for row in csv_reader:
        counter += 1
        if counter % 100 == 0:
            print 'read {} lines'.format(counter)
        timestamp = du_parser.parse(row[0])
        p_field = float(row[1])
        result = model.run({'timestamp': timestamp, predicted_field: p_field})
        result = shifter.shift(result)
        prediction = result.inferences['multiStepBestPredictions'][1]
        anomalyScore = result.inferences['anomalyScore']
        output.write(timestamp, p_field, prediction, anomalyScore)
    input_file.close()
    output.close()


def main(input_file, output_name, plot, predicted_field):
    from model_params import model_params
    model = create_model(model_params.MODEL_PARAMS, predicted_field)
    run_model(model, input_file, output_name, plot, predicted_field)


if __name__ == '__main__':
    import sys
    plot = False
    args = sys.argv[1:]
    if "--input_file" in args:
        input_file = index_col = args[args.index('--input_file') + 1]
    else:
        input_file = 'data.csv'
    if "--plot" in args:
        plot = True
    if "--output_name" in args:
        output_name = index_col = args[args.index('--output_name') + 1]
    else:
        output_name = 'data'
    main(input_file, output_name, plot)

