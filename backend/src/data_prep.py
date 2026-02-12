import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
import math
import joblib
import os
from sqlalchemy import create_engine

# Veri yolu ayarları
RAW_AIRBNB_DATA_PATH = 'data/raw/AB_NYC_2019.csv'
RAW_METRO_DATA_PATH = 'data/raw/Nyc_Subway.csv'
RAW_EVENT_DATA_PATH = 'data/raw/Nyc_Permitted_Events.csv'

try:
    df = pd.read_csv(RAW_AIRBNB_DATA_PATH)
    metro_df = pd.read_csv(RAW_METRO_DATA_PATH)
    events_df = pd.read_csv(RAW_EVENT_DATA_PATH)
except FileNotFoundError:
    print("'data/raw' klasörünü kontrol et.")
    exit() # Hata varsa dur


# haversine formülünü hesaplayan fonksiyon
def haversine(enlem_bir, boylam_bir, enlem_iki, boylam_iki):
    R = 6378 # ortalama dünya yarıçapı
    enlem_bir = math.radians(enlem_bir)
    enlem_iki = math.radians(enlem_iki)
    boylam_bir = math.radians(boylam_bir)
    boylam_iki = math.radians(boylam_iki)
    d_enlem = enlem_bir - enlem_iki
    d_boylam = boylam_bir - boylam_iki
    #haversine formülü
    hav = (math.sin(d_enlem / 2) ** 2) + (math.cos(enlem_bir) * math.cos(enlem_iki) * (math.sin(d_boylam / 2) ** 2))
    #haversine formülünün değerini açıya çevirme
    c = 2 * (math.atan2(math.sqrt(hav), math.sqrt(1 - hav)))
    #uzaklık hesabı
    distance = c * R
    return distance


# dünyanın yarı çapı (metre)
world_r = 6371000


# Times Meydanının koordinatları, bu noktaya uzaklığı hesaplayıp fiyat tahmininde kullanacağız.
new_york_times_square_lat = 40.7580
new_york_times_square_long = -73.9855


# exist_ok ile var ise ellemez yok ise bu klasörleri oluşturur.
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)




#gereksiz veri temizliği ----

df.drop(['id', 'name', 'host_id', 'host_name', 'neighbourhood'], axis=1, inplace=True) #gereksiz görülen kısımlar atıldı
df['reviews_per_month'] = df['reviews_per_month'].fillna(0) #son ayın değerlendirmelerinde null değerleri 0 ile değiştirildi
df.drop(['last_review'], axis=1, inplace=True, errors='ignore') #Son değerlendirmeler silindi



# verisetini analiz ettiğimizde 400 doların üstünde bulunan ev sayısı genel duruma aykırı düşüyor
# daha doğru bir tespit için ev miktarını 400 dolar altında bulunan evlere düşüreceğiz
df = df[df['price'] > 0]
df = df[df['price'] < 400]



#Times meydanına uzaklık stunu işlemleri!!!

# şimdi öncelikle 'times_uzaklik' isimli bir stun oluşturduk. Bu stun için kendi df'imiz içerisindeki 
# satırlardan latitude yani bulunduğumuz o anki evin enlemini ve longitude yani boylamını alıp o sıradaki ev için 
# fonksiyona gönderdik. Fonksiyone ikinci bölge olarak new york times meydanının koordinatlarını gönderdik. 
# bu sayede yukarıda yazdığımız haversine fonksiyonu teker teker her satır yani her bir ev verisi için new york times meydanına
# olan uzaklıklarını ölçüp frameimize yazacak.
df['times_uzaklik'] = df.apply(
     lambda row: haversine(row['latitude'], row['longitude'], new_york_times_square_lat, new_york_times_square_long),
    axis = 1)
#Times meydanına uzaklık stunu bitii!!!!!




# NYC Etkinlik veri setini okuduk.
events_df = pd.read_csv(RAW_EVENT_DATA_PATH)

# Öncelikle veri setindeki etkinlik tiplerine bi göz gezdirdik. Gereksiz olarak tutulan bir sürü etkinlik listelenmiş durumdadır.
unique_events = events_df['Event Type'].unique()

# needed_event_df = events_df[events_df['']]

# Gereksiz olan etkinlik listelerini çıkarttık ve ham etkinliklerimizi, gerçekten kalabalık oluşturacak
# vurucu etkinlikler, aldık.
needed_event_df = events_df[events_df['Event Type'].isin(['Special Event', 'Parade', 'Street Event', 'Religious Event', 'Press Conference', 'Street Festival', 'Athletic', 'Marathon'])]

# İlgili etkinliklerin hangi ilçelerde olduğuna baktık. 
broughts = needed_event_df['Event Borough'].unique()

# İlçelere göre ayırdık.
borough_groups = needed_event_df.groupby(['Event Borough'])

# Şimdi elimizde ilçelere göre kategori edilmiş bir data var. Bu data bize hangi ilçede ne kadar yoğunlukla etkinlik yapılıyor
# onu gösterecektir. Biz bu etkinliklerin çokluğuna göre skor ataması olarak kullanabiliriz. 
# İlgili veride XGBoost modelini kullandığımız için normalize etmeye gerek yoktur.
borough_activity_dict = borough_groups.size().to_dict()

print(borough_activity_dict)

df['neighbourhood_group'] = df['neighbourhood_group'].str.strip()

df['borough_activity_score'] = df['neighbourhood_group'].map(borough_activity_dict) 

df['borough_activity_score'] = df['borough_activity_score'].fillna(0)






# metrolar için kullandığımız veri setinden yalnızca istek üzerine belirttiğimiz stunları seçtik.
metro_df = metro_df[['Station Name', 'Entrance Latitude', 'Entrance Longitude']]
# Stunlarımız tertemiz olduğu için herhangi ek bir müdahalede bulunmamıza gerek yok

# Her bir evin en yakınındaki metroyu hesaplamak için BallTree kullanacağız. 
# BallTree algoritması radyanlar üzerinden çalışır. Fakat bizde koordinatlar bulunmakta.
# Elimizdeki koordinatları radyan cinsine dönüştürüyoruz.
metro_df_radians = np.radians(metro_df[['Entrance Latitude', 'Entrance Longitude']])


# Karar ağacımızı oluşturuyoruz. Haversine metriğini kullanmamız gerekmektedir. Dünyamızın şekline
# daha uygun bir karar yapısı oluşturacaktır. 
tree_metro_v1 = BallTree(metro_df_radians, metric='haversine')




# Karar ağacımızı kurduk evet. Şimdi de karar ağacımızın içerisinden geçirebilmek adına airbnb ev verilerimizi
# radyan cinsine çevirelim. 
df_radians = np.radians(df[['latitude', 'longitude']])


# Karar ağacından her bir evin verisini geçirdik. Karar ağacımız bize öncelikle en yakın olgunun kaç radyan uzaklıkta 
# olduğunu, sonra da hangi verinin en yakın olduğunu belirten index numarasını verecektir. 
distances, indices = tree_metro_v1.query(df_radians)


# Elimizdeki radyan cinsinden veriyi dünya yarıçapı ile çarpıp metre cinsinden veri setimize işliyoruz.
df['nearest_subway'] = distances * world_r

# Dataframe'i csv dosyasına dönüştürüyoruz ve verilen path üzerine kaydediyoruz. 
df.to_csv('data/processed/NY_With_SubwayAndEvents_vol1.csv', index=False)
print("Yeni Data kaydedildi!")

# Joblib kullanarak ağar karar yapımızı models klasörüne kaydediyoruz.
joblib.dump(tree_metro_v1, 'models/subway_tree_v1.joblib')
print("Model kaydedildi.")


joblib.dump(borough_activity_dict, 'models/borough_activity_dict.joblib')
print("Aktivite puanları kaydedildi.")


# SQL BAĞLANTI AYARLARI 
# Format: mysql+pymysql://KULLANICI:SIFRE@HOST:PORT/VERITABANI_ADI
db_connection_str = 'mysql+pymysql://root:Ybs.1453@localhost:3306/airbnb_db'

# MOTOR OLUŞTURMA
db_conn = create_engine(db_connection_str)

# Veri gönderme
try:
    df.to_sql(name='processed_nyc_data', con=db_conn, if_exists='replace', index=False)
    print("veri aktarımı başarılı")
except Exception as e:
    print("Veri aktarımı başarısız: ", e),

