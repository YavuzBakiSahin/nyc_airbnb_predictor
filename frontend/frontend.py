import streamlit as st
import requests
import os

# Başlık 
st.title("NYC Airbnb Fiyat Tahmin Uygulaması")


# Formlarımız
neighbourhood_group = st.selectbox("Bölge", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
room_type = st.selectbox("Oda Tipi", ["Entire home/apt", "Private room", "Shared room"])
minimum_nights = st.number_input("Minimum Gecelik", 1, 365)
latitude = st.number_input("Enlem", 40.48, 40.92)
longitude = st.number_input("Boylam", -74.26, -73.70)
number_of_reviews = st.number_input("Toplam Yorum Sayısı", 0, 1000)
reviews_per_month = st.number_input("Aylık Yorum Sayısı", 0.0, 50.0)
availability_365 = st.slider("Yıllık Müsaitlik", 0, 365)
calculated_host_listings_count = st.number_input("Ev Sahibinin İlan Sayısı", 1, 100)

if st.button("Tahmin!"):
    # Verimizi paketliyoruz.
    packet = {
        "neighbourhood_group": neighbourhood_group,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "availability_365": availability_365,
        "calculated_host_listings_count": calculated_host_listings_count,
        "reviews_per_month": reviews_per_month,
        "number_of_reviews": number_of_reviews,
        "latitude": latitude,
        "longitude": longitude
    }

    # API istek atma
    try:
        # Veri paketimizi backend e yolluyoruz ve dönüş bekliyoruz
        api_url = os.getenv("API_URL", "http://localhost:8000")
        full_url = f"{api_url}/predict"
        response = requests.post(full_url, json=packet)

        if response.status_code == 200:
            price = response.json()
            st.success(f"Tahmini Fiyat: {price['predicted_price_2026']} $")
            st.info(f"Metroya Uzaklık: {price['details']['metro_dist']} metre")

        else:
            st.error("Hata oluştu!")
            st.write(response.text)
            
    except Exception as e:
        st.error(f"Bağlantı Hatası: {e}")






