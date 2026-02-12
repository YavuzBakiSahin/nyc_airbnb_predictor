# 🗽 NYC Airbnb Price Predictor (Updated Edition)

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue?style=for-the-badge&logo=docker)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit)
![MySQL](https://img.shields.io/badge/MySQL-Database-4479A1?style=for-the-badge&logo=mysql)

## Proje Hakkında

Bu proje, New York City'deki Airbnb evlerinin gecelik fiyatlarını tahmin eden, uçtan uca (End-to-End) geliştirilmiş bir Makine Öğrenimi uygulamasıdır. Projemiz için kullanılan veriler 2019 yılına dayalı olmaktadır.

Tek parça bir yapının aksine **Modern Microservices Mimarisi** kullanılarak tasarlanmıştır. Veritabanı, Frontend ve Backend yapıları birbirinden izole şekilde farklı Docker Containerları üzerinde çalışır ve Docker Network ile birbirleri ile haberleşmektedirler.

## Mimari Yapı

Proje, Docker Compose ile yönetilen 3 ana servisten oluşur:

1.  **MySQL Database (`nyc_database`):** Verilerin tutulduğu katman. Başlangıçta `init.sql` scripti ile veriler otomatik olarak yüklenir ve `Docker Volumes` sayesinde veriler kalıcı hale getirilir.
2.  **Backend API (`backend_api`):** FastAPI framework'ü ile geliştirilmiştir. Eğitilmiş makine öğrenimi modelini (`.joblib`) barındırır ve tahmin isteklerini karşılar.
3.  **Frontend UI (`frontend_ui`):** Streamlit ile geliştirilmiştir. Kullanıcı dostu bir arayüz sunar ve Backend API ile iletişim kurar.

## Özellikler

**Dockerization:** ''docker-compose up'' komutu ile tüm sistem tek seferde başlar. 
**Makine Öğrenimi:** XGBoost ile eğitilmiş regresyon modeli kullanılmıştır.
**Ayrık Servisler:** Frontend ve Backend bağımsız bir şekilde çalışmaktadır. Docker Network ile birbirleri ile haberleşirler.
**Veri Kalıcılığı:** Docker Volumes kullanılarak konteyner silinse bile veri kaybı engellenmiştir.
**Otomatik Kurulum:** Veritabanı ilk açılışta backup.sql dosyasından otomatik olarak beslenir.

## Tech Stack 

Dil: Python 3.10

Orkestrasyon: Docker & Docker Compose

Backend: FastAPI, Uvicorn

Frontend: Streamlit

Veritabanı: MySQL 8.0

Veri Bilimi: Pandas, Scikit-Learn, Joblib, XGBoost

## Kurulum

Bu projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.

1. Projeyi Klonlayın
```bash
git clone [https://github.com/](https://github.com/)YavuzBakiSahin/nyc_airbnb_predictor.git
cd nyc-airbnb-predictor
```

2. .env Dosyasını Oluşturun
        - Ana dizinde .env adında bir dosya oluşturun ve veritabanı bağlantı bilgilerini girin:
            DB_ROOT_PASSWORD=gizlisifreniz
            DB_DATABASE=airbnb_db
            DB_USER=root
            DB_PASSWORD=gizlisifreniz
            DB_HOST=nyc_database
            DB_PORT=3306
            DB_NAME=airbnb_db

3. Docker ile Başlatın
Tüm servisleri inşa etmek ve başlatmak için terminalde şu komutu çalıştırın:
```bash
docker-compose up --build
```

4. Uygulamaya Erişin
Tarayıcınızda şu adrese gidin: 👉 http://localhost:8501
