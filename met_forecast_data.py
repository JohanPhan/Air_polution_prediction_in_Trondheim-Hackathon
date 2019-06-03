import json
import pandas as pd
from pandas.io.json import json_normalize
#import multiprocessing as mp
#import datetime
#from datetime import timedelta
#import re
import requests


"""
Pulling from MET's location forecast API for Trondheim's weather station "SN68860"8, Voll (lat=63.4107 lon=10.4538)
"""


def get_url(lat=63.4107, lon=10.4538):
    return f'https://api.met.no/weatherapi/locationforecast/1.9/.json?lat={lat}&lon={lon}'

def retrieve_forecasts(lat=63.4107, lon=10.4538): # Default location is Voll weather station
    response = requests.get(f'https://api.met.no/weatherapi/locationforecast/1.9/.json?lat={lat}&lon={lon}', stream=True)
    
    data = response.json()
    forecast_created = data["created"] # Record what time the forecast was created
    data = data["product"]["time"] 

    # Select the features we want for our forecast dataset
    forecasts = []
    for i in range(len(data)-1):
        line = data[i]

        if line["to"] == line["from"]: # These are the lines with the forecast info (other lines only contain precipitation forecasts)
            forecast =  {}
            forecast["timestamp"] = line["to"].replace("T", " ").replace("Z", "")
            forecast["humidity"] = line["location"]["humidity"]["value"] # percent 
            forecast["pressure"] = line["location"]["pressure"]["value"]  # hPa
            forecast["rain"] = data[i+1]["location"]["precipitation"]["value"] # millimeters of rain in the preceding hour. Several steps up to 6 hours timeframe are available
            forecast["temperature"] = line["location"]["temperature"]["value"] # C
            forecast["wind_from_direction"] = line["location"]["windDirection"]["name"] # Gets N/NW/W etc. I'm assuming that the data are the "from" direction. Degrees are also available
            
            # Encode directions (should be refactored to something faster and better)
            if forecast["wind_from_direction"] == "N":
                forecast["wind_from_direction"] = 0
            elif forecast["wind_from_direction"] == "NE":
                forecast["wind_from_direction"] = 1
            elif forecast["wind_from_direction"] == "E":
                forecast["wind_from_direction"] = 2
            elif forecast["wind_from_direction"] == "SE":
                forecast["wind_from_direction"] = 3
            elif forecast["wind_from_direction"] == "S":
                forecast["wind_from_direction"] = 4
            elif forecast["wind_from_direction"] == "SW":
                forecast["wind_from_direction"] = 5
            elif forecast["wind_from_direction"] == "W":
                forecast["wind_from_direction"] = 6
            elif forecast["wind_from_direction"] == "NW":
                forecast["wind_from_direction"] = 7
        
            forecast["wind_speed"] = line["location"]["windSpeed"]["mps"] # Name and beaufort also available
            # if "windGust" in line["location"]: # Long term forecasts don't have windGust
            #     forecast["wind_gust"] = line["location"]["windGust"]["mps"] # Unsure of usefulness. Other feature "areaMaxWindSpeed" also available
            forecast["dew_point"] = line["location"]["dewpointTemperature"]["value"] # C
            forecast["cloudiness"] = line["location"]["cloudiness"]["percent"]
            # forecast["high_clouds"] = line["location"]["highClouds"]["percent"]
            # forecast["medium_clouds"] = line["location"]["mediumClouds"]["percent"]
            # forecast["low_clouds"] = line["location"]["lowClouds"]["percent"]
            # if "fog" in line["location"]: # Long term forecasts don't have fog
            #     forecast["fog"] = line["location"]["fog"]["percent"]

            forecasts.append(forecast)

    return forecasts, forecast_created

def create_forecast_dataframe():
    forecasts, forecast_created = retrieve_forecasts()
    df = pd.DataFrame.from_dict(json_normalize(forecasts), orient='columns')

    # Currently only exporting core feature, can add more features if the models are trained on historical data
    df = df[["timestamp", "humidity", "pressure", "rain", "temperature", "wind_from_direction", "wind_speed"]]

    df.index = df["timestamp"]
    df.drop("timestamp", axis=1, inplace=True)

    return df, forecast_created



def main():
    df, forecast_created = create_forecast_dataframe()
    
    forecast_created = forecast_created.replace("Z", "").replace("T", "_").replace(":", "-")
    df.to_csv(f"../data/met_forecast_data_{forecast_created}.csv")

if __name__ == '__main__':
    main()
