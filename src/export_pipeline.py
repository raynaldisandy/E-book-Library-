"""Ekspor kandidat TF-IDF yang dapat memproses judul dan abstrak baru.

JANGAN menamai kandidat sebagai best_model tanpa validasi relevansi independen.
"""
from pathlib import Path
import json
import platform
import time
import joblib
import pandas as pd
import numpy as np
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .recommender import normalize_text


def export_candidate_tfidf(project_root: str | Path) -> Path:
    root=Path(project_root)
    book_path=root/'data'/'dataset_buku_2000.csv'
    profile_path=root/'outputs'/'profil_kompetensi_query.csv'
    if not profile_path.exists():
        profile_path=root/'data'/'profil_kerja_usulan_67.csv'
    books=pd.read_csv(book_path,dtype={'book_id':str}).fillna('')
    profiles=pd.read_csv(profile_path,dtype={'competency_id':str}).fillna('')
    if 'competency_name' not in profiles:
        raise ValueError('Profil kompetensi membutuhkan competency_name')
    if 'teks_query' not in profiles:
        if 'definisi_kerja_usulan' not in profiles:
            raise ValueError('Profil kompetensi belum menyertakan teks_query/definisi_kerja_usulan')
        profiles['teks_query']=(profiles['competency_name'].astype(str)+' '+profiles['definisi_kerja_usulan'].astype(str))
    assert books.book_id.is_unique and profiles.competency_id.is_unique
    books['teks_model']=(books['judul'].map(normalize_text)+' . '+books['abstrak'].map(normalize_text))
    profiles['teks_query']=profiles['teks_query'].map(normalize_text)
    assert (books['teks_model'].str.len()>0).all() and (profiles['teks_query'].str.len()>0).all()

    # Vocabulary dipelajari dari teks buku dan profil kompetensi (tanpa menggunakan label lama).
    documents=books.teks_model.tolist()
    queries=profiles.teks_query.tolist()
    vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=25000,sublinear_tf=True,lowercase=True,strip_accents='unicode')
    joint=vectorizer.fit_transform(documents+queries).tocsr()
    bvec=joint[:len(books)].tocsr()
    qvec=joint[len(books):].tocsr()
    folder=root/'models'/'candidates'/'tfidf'
    folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(vectorizer,folder/'vectorizer.joblib')
    sparse.save_npz(folder/'catalog_vectors.npz',bvec)
    sparse.save_npz(folder/'competency_vectors.npz',qvec)
    books.drop(columns=['teks_model']).to_csv(folder/'catalog.csv',index=False,encoding='utf-8-sig')
    profiles.to_csv(folder/'competency_profiles.csv',index=False,encoding='utf-8-sig')
    manifest={'model_name':'tfidf','model_status':'CANDIDATE_UNVALIDATED',
              'selection_status':'BELUM_MODEL_TERBAIK',
              'input_schema':{'title':'string','abstract':'string'},
              'threshold':None,'threshold_status':'BELUM_DIKALIBRASI',
              'books':len(books),'competencies':len(profiles),
              'feature_count':len(vectorizer.vocabulary_),
              'feature_config':{'ngram_range':[1,2],'max_features':25000,'sublinear_tf':True},
              'library_versions':{'sklearn':sklearn.__version__,'python':platform.python_version()},
              'notes':'Skor cosine similarity tidak sama dengan probabilitas relevansi. Profil kompetensi adalah usulan eksperimen.'}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return folder


def export_semantic_candidate(project_root: str | Path, model_name: str) -> Path:
    """Ekspor model semantic lokal + embedding untuk input buku BARU.

    Panggil hanya bila sentence-transformers dan bobot model tersedia.
    Ini memakai model pretrained yang sama sebagai encoder saat integrasi,
    bukan sekadar mengekspor CSV ranking statis.
    """
    if model_name not in ('e5', 'bge_m3'):
        raise ValueError('Hanya menerima e5 atau bge_m3')
    from sentence_transformers import SentenceTransformer
    model_ids={'e5':'intfloat/multilingual-e5-base','bge_m3':'BAAI/bge-m3'}
    root=Path(project_root)
    books=pd.read_csv(root/'data'/'dataset_buku_2000.csv',dtype={'book_id':str}).fillna('')
    profiles=pd.read_csv(root/'outputs'/'profil_kompetensi_query.csv',dtype={'competency_id':str}).fillna('')
    assert books.book_id.is_unique and profiles.competency_id.is_unique
    texts=(books['judul'].map(normalize_text)+'. '+books['abstrak'].map(normalize_text)).tolist()
    queries=profiles['teks_query'].map(normalize_text).tolist()
    if model_name=='e5':
        texts=['passage: '+s for s in texts]
        queries=['query: '+s for s in queries]
    model=SentenceTransformer(model_ids[model_name])
    book_vectors=model.encode(texts,batch_size=8,normalize_embeddings=True,show_progress_bar=True)
    competency_vectors=model.encode(queries,batch_size=8,normalize_embeddings=True,show_progress_bar=True)
    destination=root/'models'/'candidates'/model_name
    destination.mkdir(parents=True,exist_ok=True)
    model.save(str(destination/'pretrained_model'))
    np.save(destination/'catalog_vectors.npy',np.asarray(book_vectors,dtype='float32'))
    np.save(destination/'competency_vectors.npy',np.asarray(competency_vectors,dtype='float32'))
    books.to_csv(destination/'catalog.csv',index=False,encoding='utf-8-sig')
    profiles.to_csv(destination/'competency_profiles.csv',index=False,encoding='utf-8-sig')
    manifest={'model_name':model_name,'model_status':'CANDIDATE_UNVALIDATED',
              'selection_status':'BELUM_MODEL_TERBAIK',
              'pretrained_id':str((destination/'pretrained_model').resolve()),
              'original_pretrained_id':model_ids[model_name],
              'books':len(books),'competencies':len(profiles),
              'embedding_dimension':int(book_vectors.shape[1]),
              'threshold':None,'threshold_status':'BELUM_DIKALIBRASI',
              'notes':'Bobot pretrained disimpan lokal; output merupakan kandidat similarity belum tervalidasi.'}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return destination
