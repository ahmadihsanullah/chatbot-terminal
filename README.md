# Chatbot Telegram: Asisten Informasi STT Nurul Fikri

## Gambaran Proyek
Ini adalah chatbotdirancang untuk memberikan informasi komprehensif tentang STT Nurul Fikri, menggunakan teknik machine learning untuk memahami dan merespons bahasa alami.

## Fitur Utama
- 📚 Mencakup berbagai topik tentang STT Nurul Fikri
- 🧠 Menggunakan machine learning untuk klasifikasi intent


## Teknologi yang Digunakan
- Python
- scikit-learn (Machine Learning)
- TfidfVectorizer
- Klasifikasi Naive Bayes Multinomial

## Komponen Utama
- `chatbot_sklearn_training.py`: Skrip pelatihan model machine learning
- `data/intents.json`: Kumpulan data intent komprehensif
- Preprocessing teks
- Klasifikasi intent
- Pemilihan respons acak

## Cara Kerja
1. Preprocessing teks (huruf kecil, penghapusan tanda baca)
2. Ekstraksi fitur menggunakan TF-IDF
3. Klasifikasi intent menggunakan Naive Bayes
4. Generasi respons kontekstual

## Topik yang Dicakup
- Prodi Sistem Informasi
- Prodi Teknik Informatika
- Kebijakan Akademik
- Program MBKM
- Bimbingan Tugas Akhir
- Informasi Umum Kampus

## Instalasi dan Pengaturan
1. Clone repositori
2. Instal dependensi: `pip install -r requirements.txt`
4. Jalankan skrip 
```
    python chatbot_sklearn_training.py
```

## Rencana Pengembangan
- Teknik NLP yang lebih canggih
- Data latihan yang lebih beragam
- Pemahaman konteks yang lebih mendalam
