"""Analisis kandidat kompetensi berbasis skor yang dihasilkan 5 model NLP.

Setiap metode memberi skor sendiri. Ranking antar model boleh dibandingkan,
namun skor numeriknya tidak dapat dibandingkan langsung antar algoritma.
"""
from pathlib import Path
import pandas as pd
import numpy as np

MODELS = ('bm25', 'tfidf', 'fasttext', 'e5', 'bge_m3', 'bge_reranker')
REQUIRED = ('model','competency_id','book_id','skor_asli')

def read_model_scores(outputs: str | Path, model: str, book_ids, competency_ids) -> pd.DataFrame:
    if model not in MODELS:
        raise ValueError(f'Model tidak dikenal: {model}')
    p = Path(outputs)/f'skor_semua_pasangan_{model}.csv'
    if not p.exists():
        raise FileNotFoundError(f'Output {model} belum tersedia: {p}')
    df = pd.read_csv(p, dtype={'book_id':str,'competency_id':str})
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f'Kolom kurang: {missing}')
    if len(df)!=len(book_ids)*len(competency_ids):
        raise ValueError(f'{model}: jumlah pasangan {len(df)} tidak lengkap')
    if df.duplicated(['book_id','competency_id']).any():
        raise ValueError(f'{model}: pasangan book_id/competency_id duplikat')
    if set(df['book_id'])!=set(book_ids) or set(df['competency_id'])!=set(competency_ids):
        raise ValueError(f'{model}: identitas buku / kompetensi tidak cocok')
    df['skor_asli']=pd.to_numeric(df['skor_asli'],errors='coerce')
    if not np.isfinite(df['skor_asli']).all():
        raise ValueError(f'{model}: skor NaN/Inf ditemukan')
    if df['model'].nunique()!=1 or df['model'].iloc[0]!=model:
        raise ValueError(f'{model}: metadata nama model tidak sesuai')
    return df

def rank_competencies(df: pd.DataFrame, book_id: str, top_k: int=10) -> pd.DataFrame:
    if top_k<1: raise ValueError('top_k harus >0')
    candidates=df.loc[df['book_id']==str(book_id)].copy()
    if candidates.empty:raise KeyError(f'Buku tidak ada: {book_id}')
    # Urutan stabil; skor yang sama memakai ID kompetensi agar deterministik.
    return (candidates.sort_values(['skor_asli','competency_id'],ascending=[False,True])
            .head(top_k).assign(ranking=lambda x:range(1,len(x)+1)))

def rank_books(df:pd.DataFrame, competency_id:str, books:pd.DataFrame, top_k:int=10)->pd.DataFrame:
    if top_k<1:raise ValueError('top_k harus >0')
    relevant=df.loc[df['competency_id']==str(competency_id)].copy()
    if relevant.empty:raise KeyError(f'Kompetensi tidak ada: {competency_id}')
    relevant=relevant.merge(books[['book_id','judul','abstrak','tahun','author','kategori_buku']],on='book_id',how='left',validate='many_to_one')
    relevant['title_key']=relevant['judul'].fillna('').str.strip().str.casefold()
    relevant=relevant.sort_values(['skor_asli','book_id'],ascending=[False,True]).drop_duplicates('title_key').head(top_k)
    return relevant.drop(columns='title_key').assign(ranking=lambda x:range(1,len(x)+1))

def complete_judgements(judged:pd.DataFrame, competency_ids, min_books:int=20):
    """Ambil hanya buku dengan 67 pasangan yang SELURUHNYA dinilai."""
    judged=judged.drop_duplicates(['book_id','competency_id'],keep='last').copy()
    judged['relevance_grade_0_3']=pd.to_numeric(judged['relevance_grade_0_3'],errors='coerce')
    valid=judged[judged['relevance_grade_0_3'].isin([0,1,2,3])]
    wanted=set(competency_ids)
    eligible=[]
    for bid,group in valid.groupby('book_id'):
        if set(group['competency_id'])==wanted and len(group)==len(wanted): eligible.append(bid)
    return valid[valid['book_id'].isin(eligible)].copy(),len(eligible)>=min_books

def multilabel_scores(df:pd.DataFrame, judged:pd.DataFrame, book_ids:list[str], k:int=5)->dict:
    """P@K dan NDCG@K dari label INDEPENDEN pada semua 67 pasangan per buku.
    Jika label berasal dari LLM, hasil ini adalah evaluasi proxy LLM, BUKAN akurasi pakar.
    """
    from sklearn.metrics import ndcg_score
    truth=judged.set_index(['book_id','competency_id'])['relevance_grade_0_3']
    p_list=[]; ndcg_list=[]; match_at_k=[]
    for bid in book_ids:
        ranked=rank_competencies(df,bid,top_k=len(truth.loc[bid]))
        labels=np.array([int(truth.loc[(bid,c)]) for c in ranked['competency_id']],dtype=float)
        p_list.append(float((labels[:k]>=2).mean()))
        gain=2**labels-1
        discount=1/np.log2(np.arange(len(labels))+2)
        idcg=float(np.sum(np.sort(gain)[::-1][:k]*discount[:k]))
        dcg=float(np.sum(gain[:k]*discount[:k]))
        ndcg_list.append(dcg/idcg if idcg else 0.0)
        match_at_k.append(int((labels[:k]>=2).sum()))
    return {'precision_at_k':round(float(np.mean(p_list)),4),
            'ndcg_at_k':round(float(np.mean(ndcg_list)),4),
            'avg_relevant_at_k':round(float(np.mean(match_at_k)),3),
            'n_books':len(book_ids)}
