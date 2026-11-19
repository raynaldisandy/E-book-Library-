# Panduan Deployment & Integrasi REST API AI e-Library BPJS Kesehatan

Dokumen ini memuat panduan lengkap deployment dan pemanggilan REST API untuk integrasi ke portal **AI e-Library** dan sistem **Human Capital Management (HCM) BPJS Kesehatan**.

---

## 1. Menjalankan Server API di Lingkungan Lokal / Server

### Opsi A: Menggunakan Script Cepat (Windows)
Cukup klik ganda berkas:
```cmd
run_api.bat
```
Atau jalankan melalui terminal:
```powershell
.\.venv\Scripts\python api.py
```

### Opsi B: Menggunakan Script Cepat (Linux / macOS)
```bash
chmod +x run_api.sh
./run_api.sh
```

### Opsi C: Menggunakan Uvicorn Langsung
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2
```

Setelah server aktif, akses antarmuka berikut melalui browser:
- **Interactive Web Dashboard**: [http://localhost:8000/](http://localhost:8000/)
- **Dokumentasi Interaktif Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Dokumentasi Alternatif ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 2. Ringkasan Endpoint API

| Method | Endpoint | Fungsi Bisnis e-Library | Parameter Utama |
| :---: | :--- | :--- | :--- |
| `GET` | `/` | Dasbor Web Interaktif e-Library API | - |
| `GET` | `/health` | Pengecekan Kesehatan & Status Model Aktif | - |
| `POST` | `/api/v1/recommend-competencies` | **Asupan Buku Baru**: Menerima judul & abstrak, menghasilkan 1–5 kompetensi BPJS | `title`, `abstract`, `top_k` |
| `POST` | `/api/v1/recommend-books` | Mengurutkan rekomendasi buku dalam kategori yang dipilih | `competency_id`, `top_k`, `categories` (opsional) |
| `GET` | `/api/v1/competencies` | Kamus referensi 67 kompetensi kerja BPJS | - |
| `GET` | `/api/v1/catalog` | Pencarian & daftar 973 buku katalog e-library | `q`, `limit`, `offset`, `category`, `sort` (`newest`, `oldest`, `title`) |

---

## 3. Contoh Pemanggilan API (Code Snippets)

### A. Pemetaan Buku Baru ke Kompetensi (Katalogisasi Otomatis)

#### Request cURL:
```bash
curl -X POST "http://localhost:8000/api/v1/recommend-competencies" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Healthcare Clinical Risk and Safety Management",
       "abstract": "Guidelines for clinical risk assessment, incident reporting, and hospital quality assurance.",
       "top_k": 3
     }'
```

#### Response JSON:
```json
{
  "status": "success",
  "query_title": "Healthcare Clinical Risk and Safety Management",
  "total_recommendations": 1,
  "execution_time_ms": 12.45,
  "recommendations": [
    {
      "rank": 1,
      "competency_id": "C023",
      "competency_name": "Risk Management",
      "score": 0.4448,
      "confidence": "SANGAT_TINGGI",
      "model": "bge_m3",
      "model_type": "dense_biencoder",
      "is_fallback": false,
      "latency_breakdown_ms": {
        "text_cleaning_ms": 0.15,
        "embedding_encoding_ms": 11.2,
        "similarity_ranking_ms": 0.05,
        "total_ms": 12.45
      }
    }
  ]
}
```

---

### B. Rekomendasi Buku untuk Menutup Competency Gap Karyawan

#### Request cURL:
```bash
curl -X POST "http://localhost:8000/api/v1/recommend-books" \
     -H "Content-Type: application/json" \
     -d '{
       "competency_id": "C023",
       "top_k": 5
     }'
```

#### Response JSON:
```json
{
  "status": "success",
  "competency_id": "C023",
  "competency_name": "Risk Management",
  "total_books": 5,
  "query_time_ms": 3.83,
  "books": [
    {
      "rank": 1,
      "book_id": "BK00405",
      "judul": "Enterprise Risk Management in Today's World",
      "author": "Jean-Paul Louisot",
      "tahun": "2024",
      "kategori": "Business & Economics",
      "score": 0.6475,
      "skor_model": 0.6475,
      "model": "bge_m3_precomputed",
      "is_fallback": false,
      "query_latency_ms": 3.83
    }
  ]
}
```

---

### C. Pemanggilan dari Frontend JavaScript / TypeScript (Fetch API)

```javascript
// Contoh pemanggilan dari Frontend Web e-Library
async function getBookRecommendations(competencyId) {
  const response = await fetch('http://localhost:8000/api/v1/recommend-books', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      competency_id: competencyId,
      top_k: 5
    })
  });
  const data = await response.json();
  console.log("Rekomendasi Buku:", data.books);
  return data.books;
}
```

---

## 4. Deployment Menggunakan Docker

Tersedia berkas `deploy/Dockerfile` dan `deploy/docker-compose.yml` untuk deployment di server container:

```bash
# Build dan jalankan container API
docker-compose -f deploy/docker-compose.yml up --build -d

# Cek log aplikasi
docker-compose -f deploy/docker-compose.yml logs -f
```

---

## 5. Ringkasan Kesiapan Produksi
- **CORS Middleware**: Aktif (`allow_origins="*"`) sehingga frontend tidak terblokir *Same-Origin Policy*.
- **Pydantic Validation**: Setiap input divalidasi tipe dan panjang minimumnya secara otomatis.
- **Sub-5ms Latency**: Rekomendasi buku katalog beroperasi dengan vektor precomputed tanpa membebani GPU/RAM server.
- **Zero-Notebook**: Berjalan 100% mandiri pada service process Python murni.
