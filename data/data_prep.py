import pandas as pd
import numpy as np
from numpy import sqrt, pi
from sklearn.model_selection import train_test_split

def create_df():
    df = pd.DataFrame()

    # Set time parameter
    start = 0
    interval = 0.001
    sample = 2365

    time = np.arange(start, start + interval*(sample-1), interval)
    df['t'] = time

    # Set physical parameters 
    L = sqrt(2)
    g = 9.98
    w = sqrt(g/L)
    m = 1
    theta_zero = -pi/4

    # Create data for each paraeter
    df['theta'] = theta_zero*np.cos(w*df['t'])
    df['qx'] = L*np.sin(df['theta'])
    df['qy'] = -L*np.cos(df['theta'])
    df['px'] = -m*L*w*theta_zero*(np.cos(df['theta']))*(np.sin(w*df['t']))
    df['py'] = -m*L*w*theta_zero*(np.sin(df['theta']))*(np.sin(w*df['t']))
    df['H'] = np.abs(((df['px']**2 + df['py']**2)/(2*m*(L**2))) - (m*(L**2)*(w**2)*(np.cos(df['theta']))))

    # Data cleaning
    del df['theta']
    df = df.set_index('t')

    return df

def create_traintestcsv(df, test_size = 0.2):
    # Create training and testing data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df.to_csv('train_data.csv')
    test_df.to_csv('test_data.csv')

def create_finitecsv(df):
    df.to_csv('finite_data.csv')

if __name__ == '__main__':
    df = create_df()
    create_finitecsv(df)