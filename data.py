import pandas as pd
from math import pi
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
        freq = df['Freq'].dropna()
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

def training_data(run):
    train_data = pd.read_csv(f'data/clean/Run_{run}.csv')
    train_data = list(
        zip(2*pi * train_data['freq'], train_data['amp'] ** 2)
    )
    return train_data

def generate_comparison(data, weights):
    predict = predictor(weights)
    new = [[x, y, 0] for x, y in data]
    for i, point in enumerate(data):
        new[i][2] = predict(point[0])
    return new
