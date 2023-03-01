from data import *
from train import train_predictor, eval_weights, print_weights
import pandas as pd

# Values approximated from lab measurements
lab_values = {
    1: {
        'f0': 10.376,
        'w0': 4.2518,
        'beta': 0.5205
    },
    2: {
        'f0': 11.559,
        'w0': 4.260,
        'beta': 0.9170
    },
    3: {
        'f0': 13.18,
        'w0': 4.1921,
        'beta': 1.8636
    },
    4: {
        'f0': 9.1,
        'w0': 3.8792,
        'beta': 2.0647
    }
}

# Model hyperparameters to use for each run
params = {
    1: {
        'step_size':    0.003,
        'iterations':   5000,
        'batch_size':   100
    },
    2: {
        'step_size':    0.001,
        'iterations':   2000,
        'batch_size':   1
    },
    3: {
        'step_size':    0.01,
        'iterations':   1000,
        'batch_size':   1
    },
    4: {
        'step_size':    0.1,
        'iterations':   1000,
        'batch_size':   1
    }
}

# initial weights in ballpark to minimize training time
initialize = {'f0': 5, 'w0': 4, 'beta': 1}


processing_data = False
if processing_data:
    # only do once to make raw data usable
    process_data()
    exit()

final = []

record_training = True
for run in range(1, 5):
    record = None
    if record_training:
        record = f'data/history/Run_{run}.csv'

    # Do the learn
    train_data = training_data(run)
    # weights = train_predictor(
    #     train_data,
    #     **params[run],
    #     initial_w=initialize,
    #     record=record,
    #     quiet=True
    # )
    weights = dict(pd.read_csv('results.csv').iloc[run-1])

    # Generate comparison data
    comparison = compare_amplitudes(train_data, weights, lab=lab_values[run])
    comparison = pd.DataFrame(
        columns=['Frequency (rad/s)', 'Amplitude (measured)', 'Amplitude (curve fit)', 'Amplitude (experimental)'],
        data=comparison
    )
    comparison.to_csv(f'data/comparisons/amplitude/Run_{run}.csv')

    # Keep track of run results
    error = eval_weights(weights, train_data)
    final.append([run, weights['f0'], weights['w0'], weights['beta'], error])

    # Print out final weights and error
    print(f'Run {run}:')
    print_weights(weights)
    print(f'Avg squared err: {error:.05f}\n')

df = pd.DataFrame(data=final, columns=['run', 'f0', 'w0', 'beta', 'error'])
df.set_index('run')
df.to_csv('results.csv')
