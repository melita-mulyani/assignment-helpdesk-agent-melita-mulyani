
# Menggunakan Python 3.12 agar sesuai dengan environment project
FROM python:3.12-slim

# Menentukan folder kerja di dalam container
WORKDIR /app

# Menyalin daftar dependency terlebih dahulu
COPY requirements.txt .

# Menginstal dependency aplikasi
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin kode aplikasi dan knowledge base ke container
COPY app/ ./app/
COPY data/ ./data/

# Memberi informasi bahwa aplikasi menggunakan port 8000
EXPOSE 8000

# Menjalankan FastAPI menggunakan Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
