"""Penilaian proxy berbasis model AI lokal (Ollama), bukan ground truth manusia.
Tanpa Google Books API, tanpa token, dan tanpa mengirim data ke cloud.
"""
import json
from urllib.request import Request, urlopen
from urllib.error import URLError

def judge_book(competency_name, description, title, abstract, category,
               model='qwen3:8b', host='http://127.0.0.1:11434', timeout=180):
    prompt = f"""Anda penilai independen relevansi BUKU terhadap KOMPETENSI PEGAWAI.
Tentukan relevansi berdasarkan isi nyata judul, abstrak dan kategori; jangan menebak isi lain.
Skala 0=tidak terkait; 1=hubungan umum/tidak cukup; 2=mendukung topik inti; 3=sangat sesuai dan praktis.
Jika abstrak tidak cukup kuat, pilih 0 atau 1. Bukti harus berupa kutipan pendek atau parafrasa
teks yang benar-benar tersedia, jangan mengarang halaman atau klaim kemampuan yang tidak ada.
Berikan hanya JSON: {{"grade":integer_0_3,"reason":"singkat","evidence":"kutipan pendek dari teks sumber atau kosong"}}.
KOMPETENSI: {competency_name}
DESKRIPSI KEBUTUHAN: {description[:700]}
JUDUL BUKU: {title}
KATEGORI: {category}
ABSTRAK: {abstract[:2200]}"""
    payload={'model':model,'prompt':prompt,'stream':False,'format':'json',
             'options':{'temperature':0,'num_predict':220}}
    req=Request(host.rstrip('/')+'/api/generate',data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type':'application/json'},method='POST')
    with urlopen(req,timeout=timeout) as response:
        output=json.loads(response.read().decode('utf-8'))
    raw=output.get('response','').strip()
    if raw.startswith('```'):
        raw=raw.strip('`').replace('json\n','',1).strip()
    obj=json.loads(raw)
    grade=obj.get('grade')
    if type(grade) is not int or grade not in range(4):
        raise ValueError('Judge wajib memberikan grade integer 0–3')
    return {'grade':grade,'reason':str(obj.get('reason',''))[:500],
            'evidence':str(obj.get('evidence',''))[:400]}
