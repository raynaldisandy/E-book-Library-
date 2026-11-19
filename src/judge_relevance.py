"""Evaluator terpisah via Ollama lokal; tidak memanggil Google Books API.

WAJIB: output berbasis penilai AI merupakan label proxy; bukan ground truth pakar.
"""
import json, time
from pathlib import Path
import pandas as pd
import requests

RUBRIC="0: tidak berhubungan. 1: hubungan umum. 2: mendukung keterampilan. 3: spesifik mendukung kompetensi. Nilai hanya dari judul, abstrak dan deskripsi kompetensi."

def judge_pair(row, model='qwen3:8b', endpoint='http://localhost:11434/api/chat', timeout=180):
    prompt=f"""Penilaian keterkaitan materi buku dengan kompetensi kerja.
Rubrik: {RUBRIC}
Kompetensi: {row['competency_name']}
Deskripsi: {row['teks_query']}
Judul: {row['judul']}
Abstrak: {str(row['abstrak'])[:3200]}
Balas JSON SAJA dengan key grade (integer 0..3) dan reason (maksimal 30 kata)."""
    resp=requests.post(endpoint,json={'model':model,'stream':False,'format':'json',
      'options':{'temperature':0,'seed':42},
      'messages':[{'role':'user','content':prompt}]},timeout=timeout)
    resp.raise_for_status()
    item=json.loads(resp.json()['message']['content'])
    grade=int(item['grade'])
    if grade not in (0,1,2,3):raise ValueError('Grade tidak valid')
    return grade,str(item.get('reason',''))[:500]

def evaluate_pool(pool_csv,output_csv,model='qwen3:8b',max_new=None):
    pool=pd.read_csv(pool_csv,dtype={'book_id':str,'competency_id':str}).fillna('')
    output_csv=Path(output_csv)
    if output_csv.exists():
        prior=pd.read_csv(output_csv,dtype={'book_id':str,'competency_id':str})
    else:
        prior=pd.DataFrame(columns=['book_id','competency_id','relevance_grade_0_3','reason','evaluation_source','judge_model'])
    seen=set(zip(prior['book_id'].astype(str),prior['competency_id'].astype(str)))
    count=0
    for row in pool.to_dict('records'):
        key=(str(row['book_id']),str(row['competency_id']))
        if key in seen:continue
        try:grade,why=judge_pair(row,model=model)
        except Exception as err:
            print(f'Berhenti pada {key}: {type(err).__name__}: {err}')
            break
        prior.loc[len(prior)]={'book_id':key[0],'competency_id':key[1],
          'relevance_grade_0_3':grade,'reason':why,'evaluation_source':'LLM_PROXY',
          'judge_model':model}
        prior.to_csv(output_csv,index=False,encoding='utf-8-sig')
        count+=1
        if max_new is not None and count>=max_new:break
    return prior
