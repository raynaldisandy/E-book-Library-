"""Evaluasi independen untuk pasangan buku x kompetensi yang sudah diberi label.

Label 0-3 harus dibuat terpisah dari lima model utama. Jika tidak ada label,
metrik kualitas tidak dibuat. Metrik operasional tetap bisa dihitung di notebook 14.
"""
import numpy as np
import pandas as pd


def evaluate_graded_pool(top10: pd.DataFrame, labels: pd.DataFrame):
    required={'model','competency_id','book_id','ranking'}
    if not required.issubset(top10.columns):raise ValueError('Format top10 kurang kolom')
    if not {'competency_id','book_id','grade'}.issubset(labels.columns):raise ValueError('Format penilaian kurang kolom')
    unique_labels=labels.drop_duplicates(['competency_id','book_id'])
    if unique_labels.duplicated(['competency_id','book_id']).any():raise ValueError('Label ganda')
    merged=top10.merge(unique_labels[['competency_id','book_id','grade']],on=['competency_id','book_id'],how='left',validate='many_to_one')
    # Jangan perlakukan pasangan tak dinilai sebagai tidak relevan.
    coverage=merged.groupby(['model','competency_id'])['grade'].apply(lambda s:s.notna().all())
    eligible=coverage[coverage].index
    merged=merged.set_index(['model','competency_id']).loc[eligible].reset_index() if len(eligible) else merged.iloc[0:0]
    rows=[]
    for (model,competency_id),group in merged.groupby(['model','competency_id']):
        g=group.sort_values('ranking')
        if len(g)!=10:continue
        grades=np.clip(g.grade.astype(float).to_numpy(),0,3)
        relevant=grades>=2
        prec=float(relevant.mean())
        discount=1/np.log2(np.arange(2,len(grades)+2))
        dcg=float(np.dot((2**grades-1),discount))
        ideal=np.sort(grades)[::-1]
        idcg=float(np.dot((2**ideal-1),discount))
        # NDCG dihitung terhadap 10 dokumen teratas yang dinilai pada daftar ini.
        # Bukan NDCG terhadap semua kandidat relevan di katalog.
        rows.append({'model':model,'competency_id':competency_id,'precision_at_10_proxy':prec,
                     'ndcg_at_10_local_pool':dcg/idcg if idcg>0 else 0.0})
    return pd.DataFrame(rows)
