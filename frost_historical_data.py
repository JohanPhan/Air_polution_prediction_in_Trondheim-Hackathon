import json
import pandas as pd
from pandas.io.json import json_normalize
import requests


"""
Pulling from MET's public weather measurements for Trondheim's weather station "SN68860"8, Voll (lat=63.4107 lon=10.4538)
"""


def format_date(date):
    return date.replace("T", " ").replace("Z", "")[:-4]

def retrieve_measurements(start_date, end_date, station="SN68860", clientID="7133279f-5e31-415a-812f-20e93a2c7d03", secret="f066e280-093c-49b1-91ff-3c326a9c27bc"):
    url = f"https://frost.met.no/observations/v0.jsonld?sources={station}&referencetime={start_date}/{end_date}&elements="

    elements = ["relative_humidity",
                "surface_air_pressure",
                "precipitation_amount",
                "air_temperature",
                "wind_from_direction",
                "wind_speed",
                "dew_point_temperature",
                "cloud_area_fraction" # obs, octa og ikke prosent, må endres
                ]

    elements = ", ".join(elements)
    url += elements

    print(f"Fetching from source: {url}")
    
    response = requests.get(url, auth=(clientID, secret), stream=True)

    data = response.json()
    data = data["data"] 


    measurements = []

    measurements = []
    for i in range(len(data)):
        line = data[i]
        measurement =  dict()
        measurement["timestamp"] = format_date(line["referenceTime"])

        if measurement["timestamp"][-5:] == "00:00": # If whole hour measurement
            observations = line["observations"]
            for j, element in enumerate(observations):
                element_ID = element["elementId"]
                measurement[f"{element_ID}"] = observations[j]["value"]
                if element_ID == "wind_from_direction":
                    degrees = measurement["wind_from_direction"]
                    direction = ""
                    if  337.5 < degrees <= 360 or 0 <= degrees < 22.5:
                        direction = "0" # "N"
                    elif 22.5 <= degrees < 67.5:
                        direction = "1" # "NE"
                    elif 67.5 <= degrees < 112.5:
                        direction = "2" # "E"
                    elif 112.5 <= degrees < 157.5:
                        direction = "3" # "SE"
                    elif 157.5 <= degrees < 202.5:
                        direction = "4" # "S"
                    elif 202.5 <= degrees < 247.5:
                        direction = "5" # "SW"
                    elif 247.5 <= degrees < 292.5:
                        direction = "6" # "W"
                    elif 292.5 <= degrees < 337.5:
                        direction = "7" # "NW"
                    
                    measurement["wind_from_direction"] = direction
            
            measurements.append(measurement)  

    return measurements

def measurements_to_df():

    dates = {}
    for i in range(5):
        dates[i] = {"start_date": f"201{i+4}-01-01",
                    "end_date": f"201{i+5}-01-01"}
    dates[5] = {"start_date": "2019-01-01",
                "end_date": "2019-03-03"}

    all_measurements = []
 
    for line in dates:
        start_date = dates[line]["start_date"]
        end_date = dates[line]["end_date"]
        measurements = retrieve_measurements(start_date, end_date)
        all_measurements.extend(measurements)


    df = pd.DataFrame.from_dict(json_normalize(all_measurements))
    
    df.index = df["timestamp"]
    df.drop("timestamp", axis=1, inplace=True)

    return df

def main():
    df = measurements_to_df()
    df.to_csv(f"../data/frost_measurement_data_2014-2019.csv")


if __name__ == "__main__":
    main()


    
