
import pandas as pd
import numpy as np
import joblib
import math
from fastapi import FastAPI, HTTPException, status, Path
from typing import Optional
from pydantic import BaseModel
import uvicorn

# --- YARDIMCI FONKSİYONLAR -----
def haversine(enlem_bir, boylam_bir, enlem_iki, boylam_iki):
    R = 6378000 # Metre cinsinden dünya yarıçapı
    enlem_bir = enlem_bir.iloc[0]
    boylam_bir = boylam_bir.iloc[0]
    enlem_bir = math.radians(enlem_bir)
    enlem_iki = math.radians(enlem_iki)
    boylam_bir = math.radians(boylam_bir)
    boylam_iki = math.radians(boylam_iki)
    d_enlem = enlem_bir - enlem_iki
    d_boylam = boylam_bir - boylam_iki
    hav = (math.sin(d_enlem / 2) ** 2) + (math.cos(enlem_bir) * math.cos(enlem_iki) * (math.sin(d_boylam / 2) ** 2))
    c = 2 * (math.atan2(math.sqrt(hav), math.sqrt(1 - hav)))
    distance = c * R
    return distance

# Times Meydanının koordinatları, bu noktaya uzaklığı hesaplayıp fiyat tahmininde kullanacağız.
new_york_times_square_lat = 40.7580
new_york_times_square_long = -73.9855

# API kurulumu 
app = FastAPI()


# Kullanıcıdan alınacak olan bilgilerin listesi. Bu bir mimari ve 
# kullanıcı fiyatını tahmin etmek istediği bir evi bu mimariye göre tarafımıza iletmesi gerekecek.
class House(BaseModel):
    neighbourhood_group: str
    room_type: str
    minimum_nights: int
    availability_365: int
    calculated_host_listings_count: int
    reviews_per_month: float
    number_of_reviews: int
    latitude: float
    longitude: float


# Model yükleme
try:
    model = joblib.load("models/W_Nyc_SubwayAndEvents_sql_model.joblib")
    activity_points = joblib.load("models/borough_activity_dict.joblib")
    encoder = joblib.load("models/encoder.joblib")
    ball_tree = joblib.load("models/subway_tree_v1.joblib")
    print("Modeller yüklendi")
except Exception as e:
    print(f"Hata! Modeller yüklenemedi: ",e)


# Tahmin aşaması
@app.post("/predict")
def predict_price(house: House):

    data = house.dict()

    df = pd.DataFrame([data])


    #Times uzaklık hesabı
    df["times_uzaklik"] = haversine(df['latitude'], df['longitude'], new_york_times_square_lat, new_york_times_square_long)


    # Ball Tree ile en yakın metroyu hesaplicaz
    radians = np.radians(df[['latitude', 'longitude']])
    dist, ind = ball_tree.query(radians, k=1)
    df["nearest_subway"] = dist[0][0] * 6371000


    # Aktivite skoru hesaplama
    user_input_neighbourhood_group = df['neighbourhood_group']
    user_input_ng = user_input_neighbourhood_group.iloc[0].strip()
    activity = activity_points.get(user_input_ng)
    score = int(activity)
    df["borough_activity_score"] = score
    

    # Encode
    encoded_raw = encoder.transform(df[['neighbourhood_group', 'room_type']])

    df = df.drop(['neighbourhood_group', 'room_type'], axis=1)

    df = pd.concat([df, encoded_raw], axis=1)

    model_cols = model.get_booster().feature_names
    
    for col in model_cols:
        if col not in df.columns:
            df[col] = 0
            
    input_df_final = df[model_cols]


    prediction = model.predict(input_df_final)
    predicted_price = float(prediction[0])


    inflation_rate = 1.35
    final_price = predicted_price * inflation_rate



    return {
        "status": "success",
        "predicted_price_2019": round(predicted_price, 2),
        "predicted_price_2026": round(final_price, 2),
        "details": {
            "metro_dist": round(df['nearest_subway'].iloc[0], 2),
            "activity_score": score
        }
    }
#--








