"""Modul preprocessing teks dan pengayaan profil 67 kompetensi BPJS.

Mendukung normalisasi teks bilingual (Indonesia dan Inggris), pembersihan stopword,
serta pembentukan representasi teks untuk seluruh 67 kompetensi dan 973 buku.
"""
from __future__ import annotations
import re
import unicodedata
from pathlib import Path
import pandas as pd

# Stopwords bilingual sederhana untuk teks ringkasan buku dan kompetensi
ID_STOPWORDS = {
    'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua',
    'ia', 'seperti', 'jika', 'sehingga', 'kembali', 'dan', 'ini', 'karena', 'oleh',
    'saat', 'harus', 'sementara', 'setelah', 'belum', 'kami', 'sekitar', 'bagi',
    'serta', 'di', 'dari', 'telah', 'sebagai', 'masih', 'hal', 'ketika', 'adalah',
    'itu', 'dalam', 'bisa', 'bahwa', 'atau', 'hanya', 'kita', 'dengan', 'akan',
    'juga', 'ada', 'mereka', 'sudah', 'saya', 'terhadap', 'secara', 'agar', 'lain',
    'anda', 'buku', 'membahas', 'pembahasan', 'melalui', 'praktik', 'pekerjaan', 'kemampuan'
}

EN_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any',
    'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between',
    'both', 'but', 'by', 'could', 'did', 'do', 'does', 'doing', 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he',
    'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
    'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself',
    'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought',
    'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so',
    'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while',
    'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'book', 'chapter', 'author', 'presents', 'discusses', 'provides', 'guide', 'readers'
}

ALL_STOPWORDS = ID_STOPWORDS | EN_STOPWORDS


def normalize_text(text: str | None) -> str:
    """Normalisasi string: NFKC, hapus HTML, hapus whitespace berlebih."""
    if text is None or pd.isna(text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    # Hapus tag HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalisasi karakter non-ASCII aneh / quotes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    # Ganti kontrol karakter dan whitespace
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text_for_lexical(text: str | None, remove_stopwords: bool = True) -> str:
    """Membersihkan teks khusus model leksikal (BM25, TF-IDF)."""
    text = normalize_text(text).lower()
    # Hapus karakter non-alphanumeric kecuali spasi
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if len(t) > 1 and t not in ALL_STOPWORDS]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    """Tokenisasi sederhana berbasis kata untuk BM25."""
    return clean_text_for_lexical(text).split()


def load_and_enrich_competencies(data_dir: str | Path) -> pd.DataFrame:
    """Memuat 67 kompetensi dan menggabungkan profil kerja resmi / usulan."""
    data_path = Path(data_dir)
    green_file = data_path / "competencies_green.csv"
    usulan_file = data_path / "profil_kerja_usulan_67.csv"

    if not green_file.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {green_file}")
    if not usulan_file.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {usulan_file}")

    df_green = pd.read_csv(green_file, dtype=str).fillna("")
    df_usulan = pd.read_csv(usulan_file, dtype=str).fillna("")

    merged = df_green.merge(
        df_usulan,
        on="competency_id",
        how="left",
        suffixes=("", "_usulan")
    ).fillna("")

    # Bangun teks representasi bilingual kaya
    def _build_rich_text(row):
        parts = [
            str(row["competency_name"]).strip(),
            str(row.get("definisi_kerja_usulan", "")).strip(),
            str(row.get("topik_pembelajaran_usulan", "")).strip(),
            str(row.get("istilah_terkait", "")).strip()
        ]
        return ". ".join([p for p in parts if p])

    merged["teks_profil_lengkap"] = merged.apply(_build_rich_text, axis=1)
    merged["teks_profil_bersih"] = merged["teks_profil_lengkap"].apply(clean_text_for_lexical)
    merged["teks_query"] = (
        merged["competency_name"] + " " + merged.get("istilah_terkait", "")
    ).apply(normalize_text)

    # Validasi kelengkapan 67 kompetensi
    if len(merged) != 67:
        raise ValueError(f"Diharapkan 67 kompetensi, ditemukan {len(merged)}")
    if merged["competency_id"].duplicated().any():
        raise ValueError("Terdapat ID kompetensi yang duplikat")

    return merged


def load_and_prepare_books(data_dir: str | Path) -> pd.DataFrame:
    """Memuat 973 buku dan membentuk teks gabungan judul + abstrak."""
    data_path = Path(data_dir)
    books_file = data_path / "dataset_buku_2000.csv"
    if not books_file.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {books_file}")

    books = pd.read_csv(books_file, dtype=str).fillna("")
    books["judul"] = books["judul"].apply(normalize_text)
    books["abstrak"] = books["abstrak"].apply(normalize_text)
    books["teks_buku"] = books["judul"] + ". " + books["abstrak"]
    books["teks_buku_bersih"] = books["teks_buku"].apply(clean_text_for_lexical)

    # Validasi kelengkapan 973 buku
    if len(books) != 973:
        raise ValueError(f"Diharapkan 973 buku, ditemukan {len(books)}")
    if books["book_id"].duplicated().any():
        raise ValueError("Terdapat ID buku yang duplikat")

    return books
