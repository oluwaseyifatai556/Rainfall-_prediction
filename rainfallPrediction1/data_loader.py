import pandas as pd

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"

def load_and_clean_data():
    df = pd.read_csv(URL)
    df = df.dropna()

    df = df.rename(columns={
        'RainToday': 'RainYesterday',
        'RainTomorrow': 'RainToday'
    })

    df = df[df.Location.isin([
        "Melbourne",
        "MelbourneAirport",
        "Watsonia"
    ])]

    return df
