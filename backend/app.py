
import pandas as pd
import numpy as np
import joblib
import math
from fastapi import FastAPI, HTTPException, status, Path
from typing import Optional
from pydantic import BaseModel
import uvicorn

# # --- AYARLAR ---
# st.set_page_config(page_title="NYC Fiyat Tahmincisi", page_icon="🗽", layout="wide")

# # --- MODEL YÜKLEME ---
# @st.cache_resource
# def load_artifacts():
#     # Dosya yolları senin klasör yapına göre ayarlandı
#     model = joblib.load('models/W_Nyc_SubwayAndEvents_model.joblib')
#     encoder = joblib.load('models/encoder.joblib')
#     metro_tree = joblib.load('models/subway_tree_v1.joblib')
#     event_dict = joblib.load('models/borough_activity_dict.joblib')
#     return model, encoder, metro_tree, event_dict

# try:
#     model, encoder, metro_tree, event_dict = load_artifacts()
# except FileNotFoundError as e:
#     st.error(f"🚨 HATA: Dosyalar bulunamadı! {e}")
#     st.stop()

# # --- ARAYÜZ (SOL MENÜ) ---
# st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Stripe_Logo%2C_revised_2016.svg/2560px-Stripe_Logo%2C_revised_2016.svg.png", width=150) # Temsili logo
# st.sidebar.header("Ev Özellikleri")

# # Kullanıcıdan Girdileri Al
# neighbourhood = st.sidebar.selectbox("İlçe (Borough)", ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'])
# room_type = st.sidebar.selectbox("Oda Tipi", ['Entire home/apt', 'Private room', 'Shared room'])
# minimum_nights = st.sidebar.number_input("Minimum Geceleme", min_value=1, value=2)
# availability_365 = st.sidebar.slider("Yıllık Müsaitlik (Gün)", 0, 365, 150)
# calculated_host_listings_count = st.sidebar.number_input("Ev Sahibinin Diğer İlan Sayısı", min_value=1, value=1)
# reviews_per_month = st.sidebar.number_input("Aylık Yorum Sayısı (Tahmini)", value=1.0)
# number_of_reviews = st.sidebar.number_input("Toplam Yorum Sayısı", value=10)

# st.sidebar.markdown("---")
# st.sidebar.subheader("📍 Konum Seçimi")
# # Varsayılan olarak Times Meydanı koordinatları
# latitude = st.sidebar.number_input("Enlem (Latitude)", value=40.7580, format="%.5f")
# longitude = st.sidebar.number_input("Boylam (Longitude)", value=-73.9855, format="%.5f")

# # --- ANA EKRAN ---
# st.title("🗽 New York Airbnb Fiyat Tahmincisi")
# st.markdown("Yapay zeka modelimiz; metroya yakınlık, bölgesel etkinlik yoğunluğu ve ev özelliklerine göre fiyat tahmini yapar.")

# # Harita Gösterimi
# map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
# st.map(map_data, zoom=12)

# # --- TAHMİN BUTONU VE İŞLEMLER ---
# if st.button("💸 Fiyatı Tahmin Et", type="primary"):
    
#     # 1. Veriyi DataFrame'e Çevir (Ham hali)
#     input_data = pd.DataFrame({
#         'neighbourhood_group': [neighbourhood],
#         'latitude': [latitude],
#         'longitude': [longitude],
#         'room_type': [room_type],
#         'minimum_nights': [minimum_nights],
#         'number_of_reviews': [number_of_reviews],
#         'reviews_per_month': [reviews_per_month],
#         'calculated_host_listings_count': [calculated_host_listings_count],
#         'availability_365': [availability_365]
#     })

#     # 2. Times Meydanı Uzaklığı Hesapla
#     times_sq_lat, times_sq_long = 40.7580, -73.9855
#     input_data['times_uzaklik'] = haversine(latitude, longitude, times_sq_lat, times_sq_long)

#     # 3. Metro Uzaklığı Hesapla (BallTree)
#     user_coords_rad = np.radians([[latitude, longitude]])
#     dist, ind = metro_tree.query(user_coords_rad, k=1)
#     input_data['nearest_subway'] = dist[0][0] * 6371000 # Radyan -> Metre

#     # 4. Etkinlik Skoru Ekle (Sözlükten Çek)
#     # Sözlükte yoksa 0 ata
#     score = event_dict.get(neighbourhood.strip(), 0)
#     input_data['borough_activity_score'] = score

#     # 5. Encoding İşlemi (Kategorik -> Sayısal)
#     # Encoder'ı kullanarak dönüştür
#     encoded_cols = encoder.transform(input_data[['neighbourhood_group', 'room_type']])
    
#     # Ana tabloyla birleştir
#     input_data_final = pd.concat([input_data, encoded_cols], axis=1)
    
#     # Gereksiz sütunları at (Eğitimdeki sıraya uymak için)
#     input_data_final = input_data_final.drop(['neighbourhood_group', 'room_type'], axis=1)

#     # Sütun Sıralaması Garantisi (Model eğitimiyle aynı sırada olmalı)
#     # Modelin beklediği sütunları otomatik alıyoruz
#     model_cols = model.get_booster().feature_names
    
#     # Eğer eksik sütun varsa (bazen versiyon farkından olur) 0 ile doldur
#     for col in model_cols:
#         if col not in input_data_final.columns:
#             input_data_final[col] = 0
            
#     # Sadece modelin istediği sütunları, doğru sırada seç
#     input_data_final = input_data_final[model_cols]

#     # 6. Tahmin Yap
#     prediction = model.predict(input_data_final)
#     price = prediction[0]

#     # --- SONUÇ EKRANI ---
#     st.success(f"Tahmini Gecelik Fiyat: **{price:.2f} $**")
    
#     # Detay Bilgiler
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Metroya Uzaklık", f"{input_data['nearest_subway'].iloc[0]:.0f} m")
#     col2.metric("Merkeze Uzaklık", f"{input_data['times_uzaklik'].iloc[0]/1000:.1f} km")
#     col3.metric("Bölge Hareketlilik Puanı", f"{score}")


#     # ... (Önceki tahmin kodların) ...
#     price_2019 = prediction[0]
    
#     # ENFLASYON DÜZELTMESİ (2019 -> 2026)
#     # Kira artışı (%26) + Arz Kıtlığı Primi (%5-10) = ~%35
#     inflation_rate = 1.35 
#     price_2026 = price_2019 * inflation_rate

#     # --- SONUÇ EKRANI ---
#     st.success(f"Tahmini Taban Fiyat (2019 Verisi): **{price_2019:.2f} $**")
#     st.info(f"📅 2026 Enflasyon Ayarlı Tahmin: **{price_2026:.2f} $**")
    
#     # Kullanıcıya not düşelim
#     st.caption(f"Not: Bu tahmin, 2019-2026 arasındaki ~%{int((inflation_rate-1)*100)}'lik piyasa enflasyonu ve arz değişimleri eklenerek güncellenmiştir.")

# --- YARDIMCI FONKSİYONLAR ---
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








