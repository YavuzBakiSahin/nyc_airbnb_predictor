FROM python:3.10-slim


# uygulamanın bulunduğu dosyaya erişme
WORKDIR /app

# gereklilikleri indirme
COPY requirements.txt .

# gereklilikleri kopyaladık ve terminalde çalıştırılacak kodumuzu belirledik
RUN pip install --no-cache-dir -r requirements.txt


#Klasör içerisindeki her şeyi kopyalar
COPY . .

# Çalışacak portu belirleme
EXPOSE 8000


# Başlatıldığında çalışacak komut
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]