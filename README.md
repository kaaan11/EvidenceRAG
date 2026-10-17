# EvidenceRAG

EvidenceRAG, kullanıcının yüklediği dokümanlardan kanıt parçaları getirerek soruları yanıtlayan basit bir Retrieval-Augmented Generation (RAG) uygulamasıdır. Yanıt üretirken yalnızca retrieval sonucundaki parçalar kullanılır ve kullanılan chunk kaynakları yanıtla birlikte döndürülür.

## Nasıl çalışır?

```text
PDF / TXT / MD
    ↓
Metin çıkarma ve temizleme
    ↓
Overlap'li chunk'lar
    ↓
Çok dilli E5 embedding (query: / passage: ön ekleri)
    ↓
FAISS cosine similarity araması
    ↓
Relevance threshold ile kanıt seçimi
    ↓
Evidence-only prompt
    ↓
Yerel instruct LLM (Qwen2.5, chat formatı)
    ↓
Yanıt ve kaynak chunk'ları
```

PDF sayfa numaraları; dosya adı, document id, source ve chunk id ile birlikte metadata olarak korunur. Index, chunk'lar ve doküman kayıtları `data/index/store.npz` dosyasına kaydedilir ve uygulama yeniden başlatıldığında otomatik yüklenir. Dosya her değişiklikte tek adımda yerine yazılır; yazma yarıda kesilirse önceki kayıt bozulmaz.

## Kurulum

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

İlk embedding ve LLM kullanımı, ilgili Hugging Face modellerini indirir (embedding ~470 MB, LLM ~3 GB). Model GPU yoksa CPU'da float32 olarak çalışır; tipik bir dizüstü işlemcide bir cevap 10–30 saniye sürer, ilk soru model yüklemesi nedeniyle daha uzun sürer. Model adlarını veya RAG ayarlarını environment variable ile ya da proje kökündeki bir `.env` dosyasıyla değiştirebilirsiniz (`.env.example` dosyasını kopyalayıp düzenleyin; ortamda zaten tanımlı değişkenler `.env`'deki değerlerden önceliklidir):

| Değişken | Varsayılan | Açıklama |
| --- | --- | --- |
| `EMBEDDING_MODEL_NAME` | `intfloat/multilingual-e5-small` | Sentence Transformers embedding modeli (Türkçe dahil çok dilli) |
| `EMBEDDING_QUERY_PREFIX` | `query: ` | Soruya eklenen ön ek; E5 dışı modeller için boş bırakın |
| `EMBEDDING_PASSAGE_PREFIX` | `passage: ` | Chunk'lara eklenen ön ek; E5 dışı modeller için boş bırakın |
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | Chat template'i olan, `text-generation` görevini destekleyen yerel model |
| `CHUNK_SIZE` | `800` | Chunk karakter uzunluğu |
| `CHUNK_OVERLAP` | `100` | Ardışık chunk'lar arasındaki karakter overlap'i |
| `TOP_K` | `5` | Retrieval sonucundaki maksimum chunk sayısı |
| `RELEVANCE_THRESHOLD` | `0.80` | Normalize embedding'lerde cosine similarity alt sınırı |
| `MAX_NEW_TOKENS` | `256` | LLM yanıt uzunluğu sınırı |
| `MAX_UPLOAD_MB` | `20` | Yüklenebilecek en büyük doküman boyutu; aşan dosyalar `413` ile reddedilir |
| `INDEX_DIR` | `data/index` | Kalıcı index'in tutulduğu klasör (göreli yollar proje köküne göre) |

`RELEVANCE_THRESHOLD`, normalize edilmiş FAISS inner product değeridir; bu nedenle cosine similarity olarak yorumlanır. Eşik embedding modeline özgüdür: E5 modelleri skorları dar bir aralıkta (yaklaşık 0.75–0.90) üretir. 0.80 değeri `data/raw/ornek_rapor.md` üzerindeki küçük bir Türkçe soru setiyle seçildi (ilgili sorular ≥ 0.80, konu dışı sorular ≤ 0.79). Embedding modelini değiştirirseniz eşiği de yeniden ayarlayın.

## Çalıştırma

```bash
uvicorn main:app --reload
```

Sunucu başladıktan sonra web arayüzünü `http://127.0.0.1:8000/` adresinden, Swagger API ekranını ise `http://127.0.0.1:8000/docs` adresinden açabilirsiniz. Web arayüzü doküman yükleme, soru sorma, cevap gösterme ve kaynakları okunabilir kartlar halinde listeleme işlemlerini içerir.

### Sağlık kontrolü

```bash
curl http://127.0.0.1:8000/health
```

### Doküman indexleme

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "file=@./data/example.pdf"
```

Desteklenen formatlar: `.pdf`, `.txt`, `.md`. Aynı içerikteki bir dosya (adı farklı olsa bile) tekrar yüklenirse embedding hesaplanmadan `409` döner.

### Dokümanları listeleme ve silme

```bash
curl http://127.0.0.1:8000/documents
curl -X DELETE http://127.0.0.1:8000/documents/<document_id>
```

Silinen dokümanın chunk'ları index'ten çıkarılır ve artık kanıt olarak kullanılmaz. Web arayüzündeki "Yüklü dokümanlar" listesi de aynı işlemleri yapar.

### Embedding ayarlarını değiştirmek

Kayıtlı index, hangi embedding modeli ve ön eklerle oluşturulduğunu saklar. `EMBEDDING_MODEL_NAME`, `EMBEDDING_QUERY_PREFIX` veya `EMBEDDING_PASSAGE_PREFIX` değiştirilirse eski vektörler yeni modelle karşılaştırılamaz; uygulama bu durumda sessizce yanlış sonuç döndürmek yerine açık bir hatayla başlamaz. Eski ayarlara dönün ya da `data/index/store.npz` dosyasını silip dokümanları yeniden yükleyin.

### Soru sorma

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Bu dokümanın ana sonucu nedir?"}'
```

Yanıt biçimi:

```json
{
  "answer": "... [Kanıt 1]",
  "sources": [
    {
      "evidence_number": 1,
      "cited": true,
      "text": "...",
      "filename": "example.pdf",
      "document_id": "document-id",
      "page": 1,
      "source": "example.pdf",
      "chunk_id": "document-id-0",
      "score": 0.85
    }
  ],
  "verification": {
    "citations": [1],
    "invalid_citations": [],
    "unsupported_numbers": [],
    "question_only_numbers": []
  }
}
```

Kanıt threshold değerini geçmiyorsa LLM çağrılmaz ve sistem yeterli kanıt bulunamadığını açıkça bildirir. Model ret sinyali olarak yalnız `NO_EVIDENCE` döndürmelidir. Yanıtın çevresindeki whitespace temizlendikten sonra `casefold()` ile tam eşitlik aranır. ` no_evidence\n` ret sayılır; açıklama, alıntı veya `prefix_NO_EVIDENCE_suffix` içindeki marker ret sayılmaz. Ret yanıtında kaynak listesi boş ve `verification` alanı `null` olur. Diğer yanıtlarda modelin ham cevap metni korunur.

### Cevap doğrulaması

Ret olmayan cevaplar için metinsel, heuristic uyarılar üretilir (`app/rag/grounding.py`); bunlar olgusal veya semantik doğrulama değildir:

- **Atıflar:** Canonical biçim `[Kanıt N]`; `[Kanit N]` ve `[Evidence N]` alias olarak kabul edilir. Üçü de harf duyarsızdır; parantezlerin içinde çevre whitespace ve etiket ile numara arasında en az bir whitespace kabul edilir. `N` negatif olmayan ASCII rakamlarından oluşur; yalnız `1..kaynak sayısı` aralığı geçerlidir. Var olan numaralar `citations` listesine, olmayanlar `invalid_citations` listesine girer. Tanınan atıfların tamamı sayı taramasından çıkarılır; geçersiz atıf numarası da sayı alarmı olmaz. Düz `Evidence 1` veya tek başına `[1]` bu protokolün atfı değildir.
- **Sayılar:** Kanıt metinlerinde bulunan cevap sayıları için sayı uyarısı üretilmez. Yalnız soruda bulunanlar `question_only_numbers`, iki yerde de bulunmayanlar `unsupported_numbers` listesine girer. İki liste de boşken `[]` döner; sayılar cevapta görüldükleri yazımla ve sırayla, aynı yazım tekrarlanmadan sunulur. Sorudaki bir sayı kaynakta doğrulanmış sayılmaz.
- **Normalizasyon:** Noktayla üçlü gruplanan Türkçe binlikler (`18.000` / `18000`, `5.800` / `5800`) eşittir. En az iki üçlü grup içeren virgüllü veya noktalı açık binlik biçimleri (`1,234,567` / `1.234.567` / `1234567`) eşittir. İlk grup 1–3, sonraki gruplar tam 3 rakamdır. Tek virgül ondalık kabul edilir (`7,4` / `7.4`); tek nokta geçerli binlik grubuna uymuyorsa ondalıktır. Dolayısıyla `5.800` ile `5,8`, ayrıca `1,234` ile `1234` eşit değildir. Karışık veya düzensiz ayraçlar otomatik locale tahminiyle silinmez; saatlerdeki `:` ondalık noktayla eşitlenmez. İşaret, birim, bilimsel gösterim veya aritmetik sonuç için semantik sayı çözümlemesi yapılmaz.

Bu uyarılar cevabı engellemez veya yeniden üretmez. `cited=true` yalnız o kaynak numarasına atıf yapıldığını gösterir. Bir sayının Kanıt 1'de bulunup cevabın Kanıt 2'ye atıf yapması boş uyarı listeleriyle sonuçlanabilir; bu, Kanıt 2'nin iddiayı desteklediğini göstermez. Negasyon, parafraz, iddia/kaynak eşleşmesi ve genel doğruluk denetlenmez. UI, `question_only_numbers` için "yalnız soruda geçiyor; kaynakta bulunamadı" uyarısı gösterir; yeni alanı içermeyen eski API yanıtları da desteklenir.

## Testler

```bash
pytest -q
```

Testler chunk metadata'sını, embedding/vector retrieval akışını, prompt kısıtlarını, kaynak dönüşünü, yetersiz kanıt davranışını, ayarların environment'tan okunmasını ve API endpoint'lerini (yükleme doğrulaması ve boyut sınırı dahil) kontrol eder. Gerçek embedding modeli veya LLM yüklenmez; bu fonksiyonlar testte sahte fonksiyonlarla değiştirilir, bu yüzden testler birkaç saniyede biter.

## Değerlendirme

`experiments/eval/` altında Türkçe bir değerlendirme seti bulunur: birbirine benzeyen iki belediye raporu ile bir izin yönergesi ve bunlar üzerine 41 soru. Sorular dört gruptadır: cevaplanabilir (Türkçe ve İngilizce), aynı konuda olup cevabı dokümanlarda bulunmayan "tuzak" sorular ve tamamen konu dışı sorular.

```bash
python experiments/evaluate.py          # retrieval ve threshold taraması, ~1 dakika
python experiments/evaluate.py --llm    # cevapları da üretip puanlar, CPU'da ~15 dakika
```

Betik uygulamayla aynı ayarları (environment ve `.env`) kullanır; `--threshold` ve `--top-k` ile geçici olarak değiştirilebilir. Ölçülenler:

- **Retrieval:** doğru chunk'ın ilk sırada (hit@1) ve ilk `TOP_K` içinde (hit@k) bulunma oranı, MRR.
- **Threshold taraması:** her eşik için korunan cevaplanabilir soru oranı ve elenen konu dışı soru oranı. Önerilen eşik, en düşük ilgili skor ile en yüksek konu dışı skorun ortasıdır.
- **Cevaplar (`--llm`):** cevaplanabilir sorularda doğru / yanlış / reddedildi; cevaplanamayan sorularda doğru ret / uydurma; doğrulama bulguları ve cevap süresi.

Her çalıştırma `experiments/results/` altına bir JSON raporu yazar (git'e eklenmez). Model, prompt veya eşik değişikliklerinden önce ve sonra çalıştırarak karşılaştırın.

## Proje yapısı

```text
app/
  rag/
    chunking.py
    config.py
    document_loader.py
    embedding.py
    generation.py
    grounding.py
    pipeline.py
    prompt.py
    retrieval.py
    vector_store.py
data/
  raw/ornek_rapor.md
  index/            # kalıcı index (git'e eklenmez)
experiments/
  eval/
    corpus/
    dataset.json
  evaluate.py
  results/          # değerlendirme raporları (git'e eklenmez)
static/
  app.js
  index.html
  styles.css
main.py
tests/
requirements.txt
```

## Bilinen sınırlamalar

- Index tek proses için tasarlanmıştır; aynı `INDEX_DIR`'i kullanan birden fazla uvicorn worker'ı birbirinin değişikliklerini görmez.
- Her yükleme ve silmede index dosyanın tamamı yeniden yazılır; bu, birkaç bin dokümana kadar sorun olmaz.
- PDF metin çıkarma, taranmış görüntüler için OCR yapmaz.
- Şu anda yalnızca PDF, TXT ve Markdown desteklenir.
- Cevap doğrulaması atıfları ve sayıları kontrol eder; sayı içermeyen iddiaların kanıta dayanıp dayanmadığını denetlemez.
- Değerlendirme seti küçüktür (41 soru, 3 doküman); sonuçlar yön gösterir, kesin değildir.
