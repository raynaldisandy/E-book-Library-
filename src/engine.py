"""Pencocokan 67 kompetensi dengan semua buku, tanpa Google Books API."""
import re, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'; OUT=ROOT/'outputs'; MODELS=ROOT/'models'
OUT.mkdir(exist_ok=True);MODELS.mkdir(exist_ok=True)
STOP=set('the a an of and or in for to with by from at is are as on an this that buku tentang untuk pada dan yang dengan dari adalah'.split())

def tok(s):
    return [v for v in re.findall(r'[a-zA-Z]{2,}',str(s).lower()) if v not in STOP]

def load_data():
    books=pd.read_csv(DATA/'dataset_buku_2000.csv',dtype=str).fillna('')
    comps=pd.read_csv(DATA/'competencies_green.csv',dtype=str).fillna('')
    if books.book_id.duplicated().any():
        raise ValueError('book_id duplikat; periksa dataset utama')
    assert len(comps)==67, 'Daftar kompetensi bukan 67'
    # Kolom kompetensi dari scraping hanya disimpan untuk audit, TIDAK jadi filter atau label benar.
    books['teks_buku']=(books['judul']+' '+books['judul']+' '+books['abstrak']+' '+books['kategori_buku']).str.strip()
    comps['teks_query']=(comps['competency_name']+' '+comps['competency_name']+' '+comps['search_query']).str.strip()
    return books,comps

def matrix_bm25(books, comps):
    """BM25 sederhana, memakai frekuensi token; tanpa dependency rank-bm25."""
    from collections import Counter
    corpus=[tok(x) for x in books.teks_buku]
    n=len(corpus);avglen=np.mean([len(x) for x in corpus]) or 1
    df=Counter()
    for words in corpus: df.update(set(words))
    scores=np.zeros((len(comps),n))
    k1=1.5;b=0.75
    counts=[Counter(words) for words in corpus]
    for i,q in enumerate(comps.teks_query):
        for term in set(tok(q)):
            idf=np.log(1+(n-df[term]+0.5)/(df[term]+0.5))
            for j,freq in enumerate(counts):
                tf=freq[term]
                if tf:
                    denom=tf+k1*(1-b+b*len(corpus[j])/avglen)
                    scores[i,j]+=idf*tf*(k1+1)/denom
    return scores,dict(document_count=n,avg_length=avglen)

def matrix_tfidf(books, comps):
    v=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=40000,sublinear_tf=True)
    bookvec=v.fit_transform(books.teks_buku)
    queryvec=v.transform(comps.teks_query)
    return cosine_similarity(queryvec,bookvec),v

def matrix_fasttext(books,comps):
    from gensim.models import FastText
    from gensim.utils import simple_preprocess
    texts=[simple_preprocess(x) for x in books.teks_buku]
    model=FastText(sentences=texts,vector_size=80,window=5,min_count=1,workers=2,epochs=8,seed=42)
    def emb(text):
        t=simple_preprocess(text)
        return np.mean([model.wv[w] for w in t],axis=0) if t else np.zeros(model.vector_size)
    bookvec=normalize(np.stack([emb(x) for x in books.teks_buku]))
    queryvec=normalize(np.stack([emb(x) for x in comps.teks_query]))
    model.save(str(MODELS/'fasttext.model'))
    return queryvec @ bookvec.T,model

def matrix_semantic(books,comps,model_name,alias):
    # Unduh model pretrained diperlukan hanya pertama kali; bukan API Google Books.
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(model_name)
    if alias=='E5':
        booktxt=['passage: '+x for x in books.teks_buku]
        querytxt=['query: '+x for x in comps.teks_query]
    else:
        booktxt=books.teks_buku.tolist();querytxt=comps.teks_query.tolist()
    bookvec=model.encode(booktxt,batch_size=12,normalize_embeddings=True,show_progress_bar=True)
    queryvec=model.encode(querytxt,batch_size=12,normalize_embeddings=True,show_progress_bar=True)
    np.save(MODELS/(alias.lower()+'_books.npy'),bookvec)
    return queryvec@bookvec.T,model

def save_all_pairs(scores,books,comps,name,top_k=10):
    """Tidak memotong pasangan: selalu 67 x jumlah buku. Threshold untuk mapping adalah heuristik terpisah."""
    if scores.shape != (len(comps),len(books)):
        raise ValueError('Ukuran skor tidak cocok dengan dataset')
    scores=np.nan_to_num(np.asarray(scores,dtype=float),nan=0.0)
    # Mengonversi menjadi skor relatif percentile DALAM tiap kueri, bukan probabilitas benar.
    rows=[]; top=[]
    for i,c in comps.iterrows():
        s=scores[i]; order=np.argsort(-s,kind='stable')
        # Peringkat percentile 0-100 adalah peringkat relatif, BUKAN akurasi.
        relative=np.zeros(len(books))
        relative[order]=100*(len(books)-np.arange(len(books)))/len(books)
        for j in range(len(books)):
            rows.append((name,c.competency_id,c.competency_name,books.iloc[j].book_id,float(s[j]),float(relative[j])))
        # Hindari judul sama, pilih yang skor positif dan berisi isi buku
        seen=set();rank=0
        for j in order:
            b=books.iloc[j];titlekey=re.sub(r'[^a-z0-9]','',b.judul.lower())
            if not titlekey or titlekey in seen or s[j]<=0: continue
            seen.add(titlekey);rank+=1
            top.append((name,c.competency_id,c.competency_name,rank,b.book_id,b.judul,b.tahun,b.author,b.kategori_buku,b.abstrak,float(s[j]),float(relative[j])))
            if rank>=top_k:break
    pairs=pd.DataFrame(rows,columns=['model','competency_id','kompetensi','book_id','skor_asli','skor_relatif_0_100'])
    topdf=pd.DataFrame(top,columns=['model','competency_id','kompetensi','ranking','book_id','judul','tahun','author','kategori_buku','abstrak','skor_asli','skor_relatif_0_100'])
    pairs.to_csv(OUT/f'skor_semua_pasangan_{name.lower()}.csv',index=False,encoding='utf-8-sig')
    topdf.to_csv(OUT/f'top10_{name.lower()}.csv',index=False,encoding='utf-8-sig')
    return pairs,topdf

def auto_mapping(pairs,books,minimum_percentile=95,minimum_positive=True):
    """Heuristik kandidat multikompetensi: skor top 5% per kueri dan skor >0. Bukan label tervalidasi."""
    passed=pairs[pairs.skor_relatif_0_100>=minimum_percentile].copy()
    if minimum_positive: passed=passed[passed.skor_asli>0]
    mapping=passed.groupby('book_id')['kompetensi'].apply(lambda x:'; '.join(sorted(set(x)))).reset_index()
    merged=books.drop(columns=['kompetensi'],errors='ignore').merge(mapping,on='book_id',how='left')
    merged['kompetensi']=merged.kompetensi.fillna('')
    merged['jumlah_kompetensi']=merged.kompetensi.map(lambda x: len(x.split('; ')) if x else 0)
    merged['status_pemetaan']='KANDIDAT_AI_BELUM_TERVERIFIKASI'
    merged.drop(columns=['teks_buku'],errors='ignore').to_csv(OUT/'buku_multikompetensi_otomatis.csv',index=False,encoding='utf-8-sig')
    return merged

def operational_metrics(topdf,books,comps,elapsed):
    n=len(books);k=len(comps); actual=topdf.groupby('competency_id').size()
    coverage=topdf.book_id.nunique()/n if n else 0
    category_div=topdf.groupby('competency_id').kategori_buku.nunique().mean() if len(topdf) else 0
    return dict(model=topdf.model.iloc[0] if len(topdf) else 'unknown',buku_katalog=n,kompetensi=k,rekomendasi=len(topdf),fillrate_10=len(topdf)/(k*10),catalog_coverage=coverage,rata_kategori=category_div,waktu_total_detik=elapsed,waktu_per_kompetensi_detik=elapsed/k)

def overlap_table(topfiles):
    rows=[];names=list(topfiles)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            aa=topfiles[a].groupby('competency_id').book_id.apply(set).to_dict()
            bb=topfiles[b].groupby('competency_id').book_id.apply(set).to_dict()
            shared=set(aa)&set(bb)
            if shared:
                overlap=np.mean([len(aa[c]&bb[c])/10 for c in shared])
                rows.append((a,b,overlap))
    return pd.DataFrame(rows,columns=['model_a','model_b','overlap_10'])
