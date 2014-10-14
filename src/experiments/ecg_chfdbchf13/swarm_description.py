SWARM_DESCRIPTION = {
    'includedFields': [
        {
            'fieldName': 'timestamp',
            'fieldType': 'datetime',
        },
        {
            'fieldName': 'V5',
            'fieldType': 'float',
        },
    ],
    'streamDef': {
        'info': 'V5',
        'version': 1,
        'streams': [
            {
                'info': 'qtdbsel102',
                'source': 'file://data.csv',
                'columns': ['*']
            }
        ]
    },
    'inferenceType': 'TemporalAnomaly',
    'inferenceArgs': {
        'predictionSteps': [1],
        'predictedField': 'V5'
    },
    'iterationCount': 200,
    'swarmSize': 'large'
}