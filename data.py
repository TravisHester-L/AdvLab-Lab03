import pandas as pd
import numpy as np
from math import pi, sqrt
from train import predictor

col_names = ['Time', 'Angle1', 'Angle2', 'AngleAmp', 'DrivingFreq', 'Freq', 'AngleVel', 'DriverAngleVel']
raw_files = [
    'Run1_5mm.csv',
    'Run2_3mm.csv',
    'Run3_2mm.csv',
    'Run4_1mm.csv'
]

def process_data():
    for run, file in enumerate(raw_files):
        # read file into DataFrame and rename columns
        df = pd.read_csv(f'data/raw/{file}')
        col_map = {df.columns[i]: col_names[i] for i in range(8)}
        df.rename(columns=col_map, inplace=True)

        # select freq and amp columns and remove null values
        freq = 2*pi * df['Freq'].dropna()
        amp = df['AngleAmp'].dropna()

        # find the difference in column length and shave off the extra
        # beginning entries of the longest one
        diff = len(freq) - len(amp)
        if diff > 0:
            freq = freq.iloc[abs(diff):]
        elif diff < 0:
            amp = amp.iloc[abs(diff):]

        # clean up the indexes to go from 1 -> N
        freq.reset_index(drop=True, inplace=True)
        amp.reset_index(drop=True, inplace=True)

        # compose cleaned columns into DataFrame and save
        new_df = pd.DataFrame({'freq': freq, 'amp': amp})
        new_df.to_csv(f'data/clean/Run_{run+1}.csv')

def predict_phase(freq, weights):
    phase = np.arctan(2 * weights['beta'] * freq / (weights['w0']**2 - freq**2))
    return phase

def compare_phases(run, weights):
    df = pd.read_csv(f'data/phase/Run_{run}.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df.rename(columns={'Phase': 'Phase (experimental)'}, inplace=True)
    df['Phase (predicted)'] = predict_phase(df['Freq'], weights)
    df.to_csv(f'data/comparisons/phase/Run_{run}.csv')

def training_data(run):
    train_data = pd.read_csv(f'data/clean/Run_{run}.csv')
    train_data = list(
        zip(train_data['freq'], train_data['amp'] ** 2)
    )
    return train_data

def compare_amplitudes(data, weights, lab=None):
    predict_ml = predictor(weights)
    if lab is not None:
        predict_lab = predictor(lab)
        new = [[x, sqrt(y), 0, 0] for x, y in data]
    else:
        new = [[x, sqrt(y), 0] for x, y in data]

    for i, point in enumerate(data):
        new[i][2] = sqrt(predict_ml(point[0]))
        if lab is not None:
            new[i][3] = sqrt(predict_lab(point[0]))
    return new
