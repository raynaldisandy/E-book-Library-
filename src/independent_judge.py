"""Opsional: audit AI independen untuk pasangan buku-kompetensi.

Cross-encoder tidak termasuk dalam lima model retrieval yang dibandingkan.
Ini adalah penilai proksi, bukan ground truth manusia. Memerlukan download
pretrained weights pada pemakaian pertama. Tidak memakai Google Books API.
"""
from pathlib import Path
import pandas as pd


def evaluate_pairs(input_csv, output_csv, model_id='cross-encoder/ms-marco-MiniLM-L6-v2', batch_size=16):
    from sentence_transformers import CrossEncoder
    data = pd.read_csv(input_csv, dtype=str).fillna('')
    required = {'competency_id','book_id','teks_query','judul','abstrak','kategori_buku'}
    if not required.issubset(data.columns):
        raise ValueError('Kolom belum lengkap: '+str(sorted(required-set(data.columns))))
    judge = CrossEncoder(model_id)
    pairs = [
        (row.teks_query, f"{row.judul}. {row.abstrak}. Category: {row.kategori_buku}")
        for row in data.itertuples(index=False)
    ]
    # Logit relevance / ranking output: score is *not* calibrated probability.
    data['skor_judge_mentah'] = judge.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    data['judge_model'] = model_id
    Path(output_csv).parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(output_csv,index=False)
    return data
