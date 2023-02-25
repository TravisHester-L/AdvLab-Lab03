from data import training_data, process_data, generate_comparison
from train import train_predictor, eval_weights, print_weights
import pandas as pd

params = {
    1: {  # no idea why this one is so sensitive
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

initialize = {'f0': 5, 'w0': 4, 'beta': 1}

# only do once to make raw data usable
# process_data()

record_training = True
for run in range(1, 5):
    record = None
    if record_training:
        record = f'data/history/Run_{run}.csv'

    train_data = training_data(run)
    weights = train_predictor(
        train_data,
        **params[run],
        initial_w=initialize,
        record=record,
        quiet=True
    )

    comparison = generate_comparison(train_data, weights)
    comparison = pd.DataFrame(
        columns=['frequency (rad/s)', 'Amplitude (experimental)', 'Amplitude (predicted)'],
        data=comparison
    )
    comparison.to_csv(f'data/comparisons/Run_{run}.csv')

    print(f'Run {run}:')
    print_weights(weights)
    print(f'Avg squared err: {eval_weights(weights, train_data):.05f}\n')
