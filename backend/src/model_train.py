import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn.preprocessing import OneHotEncoder
import joblib
from sklearn.model_selection import GridSearchCV
import sqlalchemy
import os
from dotenv import load_dotenv

# # Veri setimizin yolu.
# DATA_PATH = 'data/processed/NY_With_SubwayAndEvents_vol1.csv'
# # Pandas ayarları.
# pd.set_option('display.max_columns', None)


# # Veri setimizi okuyoruz.
# df_airbnb = pd.read_csv(DATA_PATH)

# Değişkenleri yükle
load_dotenv()

# Değişkenleri çek
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")

# Bağlantı stringimizi otomatik olarak oluşturmamız gerekmekte
# Bağlantı için mysql+pymysql'in formatını kullandık 
db_conn_str = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

print(f"Bağlantı yapılıyor : {db_user}")


engine = sqlalchemy.create_engine(db_conn_str)

df_airbnb = pd.read_sql_table("processed_nyc_data", engine)

# OneHotEncoder ayarlamamızı yapıp objemizi tanımlıyoruz.
ohe_data = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
# OneHotEncode'layacağımız sütunları içeri atıyoruz.
ohe_data_transformed = ohe_data.fit_transform(df_airbnb[['neighbourhood_group', 'room_type']])

# OneHotEncode'lanmış olarak kullanacağımız sütunları veri setimizden siliyoruz ki üst üste binmesinler.
df_airbnb = df_airbnb.drop(['neighbourhood_group', 'room_type'], axis= 1)

# Encode'lanmış verilerimizi veri setimizin içerisine ekliyoruz.
df_airbnb = pd.concat([df_airbnb, ohe_data_transformed], axis=1)


joblib.dump(ohe_data, 'models/encoder.joblib')
print("Kaydetme başarılı")


# verileri hazırlama
x = df_airbnb.drop('price', axis=1)
y = df_airbnb['price']

# Train ve test verilerimizi bölüyoruz
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=36)


# parameter = {
#     'n_estimators' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
#     'learning_rate' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
#     'max_depth' : [1,2,3,4,5,6,7],
#     'subsample' : [0.5, 0.6, 0.7, 0.8, 0.9]
# }

# cls = GridSearchCV(
#     estimator=xgb.XGBRegressor(objective='reg:squarederror'),
#     param_grid= parameter,
#     scoring= "neg_mean_squared_error",
#     n_jobs= 1,
#     refit= True,
#     cv = 3,
#     verbose= 2
# )
# cls.fit(x_train, y_train)


# print(f"En iyisi: {cls.best_params_}")

# print(f"En iyi skor: {cls.best_score_}")


# En iyisi: {'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 1000, 'subsample': 0.7}
# En iyi skor: -2486.292236328125




 #model kurma 
# n_estimators karar ağacı sayısıdır, learning rate öğrenim değeridir.
model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=1, max_depth = 7, subsample= 0.7, colsample_bytree = 0.9)

model.fit(x_train, y_train)


#tahmin edilen verileri tahminler isimli değişkene atadık
tahminler = model.predict(x_test)
#Hata payını ölçmek için mae kullanıyoruz. Ortalam mutlak değer hata fonksiyonudur.
hata_payi = mean_absolute_error(y_test, tahminler)

print(f"Ortalama Hata (MAE): {hata_payi:.2f} $")

#R2 score için ise scikit-learn kütüphanesinin kendi fonksiyonunu kullanıyoruz
r2 = r2_score(y_test, tahminler)
print(f"Başarı Skoru (R2): {r2:.2f}")
print(f"\nİlk Hatamız: ~34.20 $")
print(f"Koordinatsız Hata: ~38.10 $")
print(f"Final Hata (Hepsi Bir Arada): {hata_payi:.2f} $")


joblib.dump(model, "models/W_Nyc_SubwayAndEvents_sql_model.joblib")
print("Model başarı ile kaydedildi.")


# Özellik Önem Düzeylerini Görselleştirme
plt.figure(figsize=(10, 8))
# Modelin içinden önem skorlarını alıyoruz
sorted_idx = model.feature_importances_.argsort()
plt.barh(x.columns[sorted_idx], model.feature_importances_[sorted_idx])
plt.xlabel("XGBoost Feature Importance")
plt.title("Hangi Özellik Fiyatı Daha Çok Etkiliyor?")
plt.show()
