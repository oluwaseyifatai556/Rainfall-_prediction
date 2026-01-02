import pandas as pd

def date_to_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'

def add_season_feature(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Season'] = df['Date'].apply(date_to_season)
    df = df.drop(columns='Date')
    return df
