# EvidenceRAG

### Yanıt teşhisi sözleşmesi

`/ask` ve doğrudan `RAGPipeline.ask` çıktısı eski `answer/sources/verification` alanlarına ek olarak `diagnostics` sürüm 1 taşır. `reason`: `answered`, `retrieval_no_evidence`, `model_no_evidence`, `input_limit`, `busy`, `model_error` veya `request_error`; hatalar HTTP durumlarını korur. `generation_invoked` gerçek generator callable girişinde true olur; loader/token bütçesi reddi false kalır.

`timings_ms` monotonic milisaniyedir: `total`, `queue`, `model_load`, `query_embedding`, `retrieval`, `generation`, `verification`. Çalışmayan aşama null, çalışan sıfır süre 0'dır. Retrieval query embedding'i, embedding de soğuk yüklemeyi kapsayabilir; alt süreler toplam request'e eklenmez. Generation yalnız callable süresidir; load ve token kontrolünü kapsamaz. Total yanıtın ağdan istemciye ulaşmasını ölçmez. Queue reddi/iptali ve çalışan işte istemci iptali `admission_state` ile ayrılır; iptal native işi sonlandırmaz.

Rastgele `request_id`, T06 model/tokenizer/index kimliği, prompt/source SHA ve runtime sürümleri içerik saklamadan teşhis sağlar. Statik kaynak/sürüm bilgisi proses başına cache'lenir; sorgu başına shell çalışmaz. Varsayılan log yalnız request ID, reason, invocation, admission state ve süreleri içerir; soru/cevap/belge/anahtar loglanmaz. Bu alanlar semantik doğruluk veya gerçek model performansı iddiası değildir.

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

PDF sayfa numaraları; dosya adı, document id, source ve chunk id ile birlikte metadata olarak korunur. Index, chunk'lar ve doküman kayıtları `data/index/store.npz` dosyasına kaydedilir ve uygulama yeniden başlatıldığında otomatik yüklenir. Yeni candidate state kaydedilip replace edilmeden canlı index'e yayımlanmaz. Commit öncesi kayıt hatası eski RAM/disk durumunu korur; commit sonrası durability belirsizliği ayrı hata olarak bildirilir. Bu sözleşme ölçülmüş power-loss dayanıklılığı değildir.

## Kurulum

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --require-hashes --only-binary=:all: -r requirements.lock
```

Lock hedefi Windows x64 / CPython 3.12'dir; temiz kurulum Windows 11/Python 3.12.10 üzerinde doğrulanır. Runtime lock 53, test lock 59 çözülmüş dağıtımın gerçek wheel SHA-256 değerlerini taşır. `requirements.txt` runtime doğrudan girdisi, `requirements-dev.txt` ona ek pytest/httpx2 test girdisidir; doğrudan sürümler korunur. Linux/macOS/ARM için bu tek-platform hash dosyaları doğrulanmış kurulum iddiası taşımaz. Kullanılan producer pip 25.0.1'dir; model ağırlıkları lock'a dahil değildir.

İlk gerçek embedding/LLM kullanımı cache yoksa pinli Hugging Face artifact'ını indirebilir; test lane'leri offline'dır ve ağırlık indirmez. Model GPU yoksa CPU float32 kullanır; gerçek inference gecikmesi/kapasitesi bu teslimde ölçülmedi. Model adlarını veya RAG ayarlarını environment variable ile ya da proje kökündeki `.env` dosyasıyla değiştirebilirsiniz (`.env.example` dosyasını kopyalayıp düzenleyin; ortamda zaten tanımlı değerler önceliklidir):

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

### Girdi ve store bütçeleri

Bu sınırlar yerel, tek sahibi olan ve tek prosesli kullanım için değiştirilebilir başlangıç politikasıdır; ölçülmüş güvenli kapasite veya gecikme garantisi değildir. Pozitif tam sayı gerekir; chunk overlap değeri `0 <= CHUNK_OVERLAP < CHUNK_SIZE` olmalıdır. Geçersiz ayarlar başlangıçta hata verir.

| Ayar | Varsayılan | Kapsam / birim |
| --- | --- | --- |
| `MAX_UPLOAD_MB` | `20` | Dosya içeriği, MiB (`1 MiB = 1.048.576 bayt`); eski ayar korunur |
| `MAX_UPLOAD_BYTES` | `MAX_UPLOAD_MB * 1.048.576` | İsteğe bağlı kesin bayt sınırı; verilirse MiB ayarını geçersiz kılar |
| `MAX_HTTP_BODY_BYTES` | `21037056` | POST `/documents` toplam gövdesi: 20 MiB + 64 KiB multipart framing/başlık payı |
| `MAX_JSON_BODY_BYTES` | `32768` | POST `/ask` toplam JSON gövdesi, bayt |
| `MAX_MULTIPART_HEADER_BYTES` | `8192` | Parça başına başlık adları/değerleri ve normalize `: ` / CRLF ayraçları için bayt bütçesi |
| `MAX_FILENAME_BYTES` | `512` | Dosya adının UTF-8 bayt uzunluğu |
| `MAX_QUESTION_CHARS` | `4000` | Soru karakterleri, çevre boşlukları dahil; token sınırı değildir |
| `MAX_PDF_PAGES` | `500` | PDF başına sayfa |
| `MAX_TEXT_CHARS` | `2000000` | Belgedeki bütün sayfalardan çıkarılan toplam metin karakteri |
| `MAX_DOCUMENT_CHUNKS` | `2000` | Belge başına üretilen chunk |
| `MAX_STORE_CHUNKS` | `20000` | Store toplam chunk sayısı |
| `MAX_STORE_TEXT_BYTES` | `67108864` | Store'daki chunk metinlerinin toplam UTF-8 baytı (64 MiB); overlap tekrarları da sayılır |

Upload yalnız `file` adlı tek dosya parçası kabul eder; ek dosya veya form alanı reddedilir. Dosya baytlarıyla toplam HTTP gövdesi ayrı sınırlardır. Bilinen aşırı `Content-Length` parser başlamadan reddedilir. Başlık yoksa veya gerçeği yansıtmıyorsa ASGI `receive` üzerinden sayılan toplam kullanılır; bütçeyi aşan chunk parser'a verilmez ve başka body chunk'ı istenmez. Limit içinde alınan önceki parçalar parse edilmiş veya diske spool edilmiş olabilir. Kısmi, tamamlanmamış ve başarısız upload kaynakları kapatılır. Bu sayaç sunucunun önceden belleğe aldığı tek ASGI chunk'ının boyutunu sınırlamaz; genel HTTP sunucu/header veya proxy sınırlarının yerine geçmez.

TXT/MD en çok metin limiti + 1 karakter okunarak kontrol edilir. PDF sayfa sayısı çıkarımdan önce, toplam metin ise her sayfa çıkarımından sonra kontrol edilir. Pypdf decoder korumaları değiştirilmez. `extract_text()` içindeki tepe CPU/RAM, page-tree dolaşımı ve dosya açma maliyeti için sert sınır sağlanmaz; thread timeout'u veya ayrı extraction prosesi eklenmemiştir. Chunk üretimi limitte durur; tüm sınırsız chunk listesi önce üretilmez. Erken dosya/metin/sayfa/chunk reddi embedding veya store mutasyonu başlatmaz.

Store kotası embedding öncesi kontrol edilir, ardından mutasyon kilidi altında yeniden denetlenir; yarış nedeniyle son kontrolde ret olursa embedding zaten yapılmış olabilir, ancak reddedilen belge RAM/disk durumuna eklenmez. 64 MiB yeni toplam metin başlangıç bütçesidir; vektörler, metadata, NPZ veya proses RSS boyutunu ölçmez. Mevcut bir index yeni kotanın üstündeyse sessizce silinmez/rebuild edilmez; okuma/silme kullanılabilir, kota içi duruma dönene kadar yeni ekleme reddedilir. Kayıt hatalarında transaction/rollback garantisi bu limit kontrolünden ayrı bir konudur.

HTTP hata biçimi `detail` alanını korur: gövde/dosya/metin/sayfa/chunk/store ve başlık/filename aşımı `413`; soru karakter sınırı API modelinde `422`; boş/geçersiz belge veya fazladan multipart parça `400`; duplicate içerik `409`. Doğrudan `pipeline.ask/index_file` çağrısı limit aşımında `InputLimitError` yükseltir. Hiçbir limit sessiz kırpma yapmaz. Upload MiB veya bayt ayarını artırırken ayrı HTTP gövde bütçesi de bilinçli seçilmelidir. Prompt token bütçesi ve aktif/bekleyen model işlerinin sınırları bu görevin kapsamında değildir.

`RELEVANCE_THRESHOLD`, normalize edilmiş FAISS inner product değeridir; bu nedenle cosine similarity olarak yorumlanır. Eşik embedding modeline özgüdür: E5 modelleri skorları dar bir aralıkta (yaklaşık 0.75–0.90) üretir. 0.80 değeri `data/raw/ornek_rapor.md` üzerindeki küçük bir Türkçe soru setiyle seçildi (ilgili sorular ≥ 0.80, konu dışı sorular ≤ 0.79). Embedding modelini değiştirirseniz eşiği de yeniden ayarlayın.

## Çalıştırma

```bash
uvicorn main:app --host 127.0.0.1 --no-proxy-headers --reload
```

Sunucu başladıktan sonra web arayüzünü `http://127.0.0.1:8000/` adresinden, Swagger API ekranını ise `http://127.0.0.1:8000/docs` adresinden açabilirsiniz. Web arayüzü doküman yükleme, soru sorma, cevap gösterme ve kaynakları okunabilir kartlar halinde listeleme işlemlerini içerir.

### Ağır model işleri ve readiness

LLM generation ve embedding inference ayrıca kendi model-kullanım mutex'ini çağrı gerçekten dönene kadar tutar; global kapasiteyi yükseltmek aynı adapter nesnesinin eşzamanlı kullanımına izin vermez.

`MAX_ACTIVE_HEAVY_JOBS=1` generation, query/index embedding ve cold loader için ortak proses-içi kapasitedir. Ask/index pipeline işleri bu iznin sahibidir; alt çağrılar yeniden izin beklemez. Doğrudan loader/embedding/generation çağrıları da aynı sınırı kullanır. Kilit sırası kapasite → loader cache → kısa store işlemleridir; store kilidini tutarak kapasite beklenmez.

Bekleyen toplam iş `MAX_WAITING_HEAVY_JOBS=2` ile sınırlıdır. Global FIFO içinde ask/index/diğer model işi sayıları ayrıca `ASK_QUEUE_CAPACITY=2`, `INDEX_QUEUE_CAPACITY=2`, `MODEL_QUEUE_CAPACITY=2` ile sınırlandırılır. `HEAVY_WAIT_SECONDS=30` yalnız kuyruk bekleme süresidir; dolu/kapanan kuyruk veya süresi dolan bekleme `503`, `Retry-After: 1` ve açıklayıcı `detail` verir. Async admission ortak threadpool'u bloke etmez; native işler ayrı, kapasite kadar worker'ı olan executor'da çalışır. Health/belge listesi ağır kuyruğa girmez.

İstemci iptali kuyruktaysa iş çıkarılır. İş başladıysa thread gerçekten bitene kadar hem global izin hem isteğe ait upload dosyaları korunur; iptal bu bekleyişten sonra bildirilir. Hata izin sızıntısı bırakmaz. Thread için sert sonlandırma/gerçek inference timeout'u yoktur; takılan native iş proses sonlanana dek sürebilir. Bu sınırlar RSS/p95 veya gösterilmemiş KV-cache arızalarının giderildiği garantisi değildir.

Default yükleme lazy'dir: startup model indirmez; `/health` içindeki `model_readiness: lazy` modelin yüklü olduğunu söylemez. `PRELOAD_MODELS=true` açık opt-in'dir; lifecycle loading/ready/failed durumunu izler, preload hatası startup'ı durdurur. Cache aynı model+revision anahtarı için tek başarılı kurucu yayınlar; ilk hata cache'e yazılmaz ve tekrar denenebilir. Revision kimlikleri ayrı model kimliği görevinin konusudur; burada SHA uydurulmaz.

Gerçek generation öncesinde tokenizer'ın `apply_chat_template` çıktısı bütün system/soru/kanıt tokenlarını ve `MAX_NEW_TOKENS` cevap rezervini kapsayarak gerçek model/tokenizer context sınırıyla karşılaştırılır. Aşım `InputLimitError` / HTTP `413` verir; soru/kanıt sessiz kesilmez. Query embedding bu kontrolden önce yapılmış olabilir. Char limiti token limiti değildir. İsteğe bağlı pozitif `TORCH_NUM_THREADS` ilk model yüklemesinden önce bir kez uygulanır; belirtilmezse host env/default thread ayarları değiştirilmez.

### Yerel erişim sınırı

`ACCESS_PROFILE=local` varsayılan ve desteklenen tek profildir. `shared`, `public` veya başka bir değer başlangıçta hata verir; bir API key eklemek bunları etkinleştirmez. Public/LAN yayın, reverse proxy ve çok kullanıcı/tenant izolasyonu bu sürümde desteklenmez. Aynı makinedeki diğer proseslere karşı kullanıcı kimlik doğrulaması sağlanmaz; bütün yerel istemciler aynı store'u kullanır.

Launcher loopback'e bağlanır ve proxy başlıklarından peer türetmez. Uygulama ayrıca her HTTP isteğinde gerçek ASGI peer'inin loopback IP olmasını ve tek bir geçerli Host başlığını ister. Host allowlist `localhost`, `127.0.0.1`, köşeli parantezli `[::1]` ve bunların 1–65535 port varyantlarıdır. IPv6 kullanacaksanız aynı launcher'da `--host ::1` seçip `http://[::1]:8000/` açabilirsiniz. `localhost.evil.example`, userinfo, eksik/çoklu Host veya bozuk port kabul edilmez. Host kontrolü socket binding veya auth yerine geçmez.

POST/DELETE gibi mutasyonlarda Origin varsa scheme/host/etkin port olarak isteğin origin'i ile tam eşleşmelidir. `null`, başka port/host, birden fazla Origin veya URL path'i içeren Origin reddedilir. `localhost` ile `127.0.0.1` aynı origin sayılmaz. Origin göndermeyen doğrudan yerel CLI istemcisi kabul edilir; örneğin `curl http://127.0.0.1:8000/health` ve yukarıdaki JSON/multipart örnekleri çalışır. Origin taklit edilebilir; bu kontrol auth değildir. CORS wildcard eklenmez.

`Forwarded`, `X-Forwarded-For/Host/Proto/Port` ve `X-Real-IP` başlıkları değerlerine güvenilmeden reddedilir. Uzak peer, sahte localhost Host/Origin ile yerel kabul edilmez. Başlıkları kaldıran yerel bir reverse proxy'nin arkasındaki gerçek istemciyi uygulama ayırt edemez; bu nedenle proxy arkasında yerel erişim garantisi verilmez ve o dağıtım desteklenmez.

Host/proxy hataları `400`, uzak peer veya uygun olmayan mutasyon Origin'i `403` ve mevcut `detail` biçimiyle döner; kontrol body parser, spool ve model/store işinden önce çalışır. `/`, `/static`, `/docs`, `/openapi.json`, `/health` ve belge/soru uçları aynı peer/Host sınırına tabidir. GET/HEAD/OPTIONS için Origin şartı yoktur. `/docs` yerelde açıktır; onu kapatmak erişim korumasının yerine geçmez. UI doğrulama hatalarını metin olarak gösterir; anahtar akışı veya browser storage eklenmemiştir.

Belge metni ve dosya adı prompt'ta JSON verisi olarak serileştirilir; yeni satır, alıntı, kontrol ve `<...>` rol/delimiter işaretleri escape edilir. Soru aynı user rolünde ayrı serileştirilir; system rolü yalnız uygulama protokolünü taşır. Kaynak komutları atıf/ret protokolünü değiştiremez şeklinde açık sistem talimatı vardır. Bu, gerçek modelin kötü niyetli talimata uymayacağını kanıtlamaz; NO_EVIDENCE tam yanıt sözleşmesi ve semantik doğrulama sınırlamaları korunur. Dar serializer/DOM-stub testleri genel prompt-injection veya XSS denetimi değildir.

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

### Kayıt işlemi ve servis lease'i

Servis kalıcı indeksi modül importunda yüklemez; lease alındıktan sonra son committed dosyayı yeniden okur ve ancak başarılı yükleme/preload sonrası istek kabul eder. Böylece önceki servisin import ile startup arasında tamamladığı kayıtlar korunur. Bozuk/uyumsuz dosyada startup başarısız olur ve lease bırakılır; dosya değiştirilmez. Standalone constructor varsayılan otomatik okumasını korur.

Add/delete aynı mutasyon kilidi altında candidate FAISS index'i, chunk listesi ve belge kayıtlarını oluşturur; quota/duplicate kontrolleri aynı kritik bölümdedir. Candidate hedefle aynı dizindeki benzersiz temporary NPZ'ye yazılır, file flush/fsync yapılır; `os.replace` görünürlük commit noktasıdır. Sonrasında tek state pointer'ı RAM'e yayımlanır. Açma/serialization/yazma/file-fsync/replace öncesi hata `not_committed` durumudur; eski RAM, arama, duplicate kontrolü ve reload korunur. Geçici dosya normal başarı/hata yollarında temizlenir; hedef dosya recovery bahanesiyle silinmez.

Replace sonrası directory sync hatası `durability_uncertain` verir: işlem görünür olabilir; temiz rollback veya normal başarı ack'i sayılmaz. API `500` yanıtında string `detail`, `operation_id` ve `commit_status` döndürür. Kör retry yapmadan belge listesini ve `/health.store_operation_id` değerini kontrol edin. Son operation ID NPZ metadata'sında da saklanır. Windows'ta file fsync uygulanır, POSIX directory-fsync uygulanmaz; POSIX yolunda directory fsync kodu vardır fakat bu Windows koşusunda doğrulanmadı. Elektrik kesintisi ve tüm dosya sistemleri için dayanıklılık garantisi yoktur.

Servis startup'ında ilk mutable kullanım öncesi index dizinindeki `.writer.lock` için OS lease alınır. Windows byte-range `msvcrt` kilidi; POSIX'te `fcntl.flock` kullanılır. Lockfile'ın varlığı veya PID sahiplik kanıtı değildir; process öldüğünde OS kilidi bırakır. Aynı dizine ikinci servis ilk mutasyondan önce hata verir; farklı dizinler bağımsızdır. Uvicorn `--workers 1` kullanın; aynı index'e ikinci reload/worker eşzamanlı yazıcı olarak başlamaz.

Shutdown yeni model/mutasyon işlerini reddeder; delete dahil başlamış native işler drain edilir. Lifespan/istemci iptali OS lease'i erken bırakmaz: ayrı watcher gerçek işler ve bekleyen ticket'lar bitene kadar kilidi tutar. Drain edilemeyen native iş varsa lease süreç gerçekten bitene kadar tutulabilir; sert thread sonlandırması yoktur. Servis lifespan'ını devre dışı bırakmak desteklenmez. Standalone `VectorStore` constructor'ları lease almaz; read/reload ve in-memory nesneler sahte çakışma yaşamaz. Standalone kalıcı yazıcılar için lease kullanımı çağıranın sorumluluğudur; bu mekanizma kötü niyetli yerel prosese karşı dosya ACL/auth yerine geçmez.

Doğrulanan platform Windows 11 x64 / Python 3.12.10'dur: native FAISS reload, I/O fault matrisi, iki gerçek prosesin OS kilidi, process kill sonrası release ve native iş + client/shutdown iptal testi çalıştırıldı. POSIX/NFS/ağ dosya sistemi ve gerçek power-loss testleri çalıştırılmadı.

### Embedding ayarlarını değiştirmek

Kayıtlı index, sürümlü embedding model/revision, aynı artifact tokenizer kimliği, prefix, L2 normalization, beklenen/etkili dimension ve ingestion/chunking kimliğini saklar. Yeni store formatı `2`'dir. `models.lock.json` resmî Hugging Face metadata'sından gözlenen gerçek immutable E5/Qwen commit'lerini, kaynak URL'lerini ve config hash'lerini içerir; uygulama Git commit'i model revision diye kullanılmaz. Default model adları korunur. `EMBEDDING_MODEL_REVISION` ve `LLM_MODEL_REVISION` ayrı 40 karakterlik lowercase SHA ayarlarıdır; custom model deposu için explicit revision gerekir. Tokenizer aynı repository/revision'a sabitlenir; branch/tag pin sayılmaz. Loader ve single-flight cache aynı etkili kimliği kullanır.

`Settings.model_identity()` deterministik model/tokenizer ve index kimliğini, `embedding_signature()` yalnız vektör üretimini etkileyen kimliği verir. `INGESTION_VERSION=1`, `CHUNKER_STRATEGY=legacy_character`, size/overlap imzaya dahildir. Yalnız LLM revision değişimi embedding index'ini reddetmez. Gerçek embedding adapter'ı model yüklendikten sonra dimension/sequence metadata'sını ve encode çıktısının boyutunu denetler; store ayrıca NPZ dimension metadata'sını matrise karşı kontrol eder. Runtime çıktısı startup'ta ölçülmüş gibi sunulmaz. Manifestteki MIT/Apache-2.0 bilgileri model artifact kartlarına aittir; repository kaynak kodu lisansı seçilmemiştir.

### Eski index'ten açık geçiş ve rollback

Revision'sız/eski format NPZ yeni SHA ile etiketlenmez; dosya değiştirilmeden açık hata verilir. Eski dosyayı silmeyin veya startup rebuild beklemeyin. Servisi durdurun, eski kod/ayarları ve index dizinini yedekleyin; yeni ayrı `INDEX_DIR` ile özgün belgelerden reindex yapın. Windows PowerShell örneği:

```powershell
# Service stopped; choose new, unused backup/index directory names.
Copy-Item -LiteralPath data/index -Destination data/index-backup -Recurse
$env:INDEX_DIR='data/index-v2'
uvicorn main:app --host 127.0.0.1 --no-proxy-headers
# Re-upload the originals explicitly, then verify /documents and /health.
```

Özgün upload dosyaları API işleminden sonra silinir; normalized NPZ chunk'larından kayıpsız PDF/Markdown geri üretimi yapılmaz. Orijinaller yoksa reindex somut olarak engellidir; eski dizini koruyup orijinalleri temin edin. Otomatik uzun süre belge saklama eklenmedi. Rollback için servisi durdurun, eski kod ve eski model/ingestion ayarlarıyla `INDEX_DIR`'i korunmuş eski dizine/backup'a döndürün; v2 dosyasını eski kodun okuyacağını varsaymayın. Yeni dizini eski dosyanın üstüne kopyalamayın. Küçük sentetik vektör/gerçek FAISS migration ve eski T05 okuyucusuyla rollback doğrulandı; gerçek-model/gerçek kullanıcı verisi migration'ı çalıştırılmadı.

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
python -m pip install --require-hashes --only-binary=:all: -r requirements-dev.lock
python -m pytest -q -p no:cacheprovider -m "model_free and not real_model"
python -m pytest -q -p no:cacheprovider -m "native and not real_model"
python experiments/evaluate.py --preflight-only --output eval-preflight.json
```

Node 24 PATH üzerinde bulunmalıdır; UI testleri küçük DOM stub'ı çalıştırır, gerçek tarayıcı denetimi değildir. `EVIDENCERAG_NODE_BINARY` açık Node executable seçebilir. `tests/conftest.py` model hub erişimini offline tutar. Model-free lane algoritma/adapter kontrolleri, native lane gerçek FAISS/framework/dosya/proses kilidi ile fake model entegrasyonudur. Toplam kontrol için `python -m pytest -q -p no:cacheprovider -m "not real_model"` kullanılabilir. Gerçek modellerle cache-only lane ancak açık `EVIDENCERAG_REAL_MODEL_TESTS=1` seçimi ve hazır immutable cache ile çalıştırılır; ağırlık indirme ayrı bütçe kararıdır.

Enjekte edilen `generate_answer(..., text_generator=callable)` test protokolü primitive `max_new_tokens`/`do_sample` argümanlarını kullanır ve Transformers/Torch gerektirmez. Üretim yolu gerçek `GenerationConfig`, revision, tokenizer bütçesi ve scheduling davranışını korur; eksik kütüphanede fake başarılı yanıt üretmez. Bu ayrım boş stdlib ortamında ayrıca doğrulanır.

`.github/workflows/tests.yml` aynı kurulum/test/preflight komutlarını Windows/Python 3.12.10/Node 24.19.0 üzerinde çalıştırır; actions checkout/setup-python/setup-node resmi repo tag'lerinden gözlenen immutable SHA'lara bağlıdır. Doğrulanan checkout `core.autocrlf=true` kullanır: legacy manifest dataset'in gerçek dosya byte hash'ini taşır; CI ephemeral runner'ında aynı newline politikası checkout öncesi açık ayarlanır. Farklı byte/newline temsili farklı dataset kimliğidir ve eski manifest sessizce yeniden etiketlenmez. Workflow dosyasının eklenmesi uzak CI başarısı değildir; PR kontrolleri ayrıca doğrulanmalıdır. Lisans seçimi hak sahibini bekler; otomatik LICENSE eklenmez.

Lock güncellemesi için desteklenen temiz Windows/Python ortamında `python -m pip install --dry-run --ignore-installed --only-binary=:all: -r requirements.txt --report runtime-resolver.json` ve dev girdisiyle aynı komut kullanılır. Her raporun `install` kayıtlarındaki gerçek `metadata.name==metadata.version` ve `download_info.archive_info.hashes.sha256` değerlerinden ilgili lock yeniden üretilir; ardından hash zorunlu temiz kurulum, iki test lane'i ve native import kontrolü tekrar yapılır. Resolver raporları ve üretici script bu görevin dış teslim paketinde kayıtlıdır. Sadece sürüm listesini değiştirmek doğrulama sayılmaz.

## Değerlendirme

### Yapı koruyan chunker (opt-in)

Varsayılan `CHUNKER_STRATEGY=legacy_character`, `INGESTION_VERSION=1` korunur; eski fixture çıktıları değişmez. Yeni kurulum/ayrı reindex için `CHUNKER_STRATEGY=structured_v2`, `INGESTION_VERSION=2` ve yeni bir `INDEX_DIR` birlikte seçilir. Strateji/version/size/overlap T06 index imzasına girer; eski dizini yeni stratejiyle açmak kontrollü uyumsuzluk hatası verir. Yedek + özgün belgelerden yeni dizine reindex + eski kod/ayar/dizine dönüş sözleşmesi geçerlidir; otomatik silme/rebuild yoktur.

Yeni strateji ATX Markdown başlıkları, satır/paragraf, tablo ve liste ilişkilerini bütçe içinde korur. Metadata `heading_path`, `char_start`, `char_end`, `source_sha256`, `strategy`, `ingestion_version`, `normalization_version`, `offset_unit` ve `source_representation` içerir. Span `[start,end)` Unicode karakter aralığıdır: TXT/MD için UTF-8 metnin CRLF/CR → LF temsilinde; PDF için extractor'ın tek sayfa LF metninde. PDF byte offset veya görsel bbox değildir. Başlık yolu metadata'dır; gösterilen chunk kaynak span'ıdır, embedding girdisi aynı metin + mevcut passage prefix'idir. Yeni provenance alanları API kaynaklarında eklemeli bulunur; hash/adapter ayrıntıları UI akışına zorunlu eklenmez.

Karakter bütçesi tokenizer bütçesi değildir; embedding adapter/modelinin kendi token tavanı uygulanabilir. Token sayısı ve truncation'ın kalite etkisi bu fixture koşusunda ölçülmedi. Kaynak span'ı, vektörün bütün karakterleri semantik olarak temsil ettiğini kanıtlamaz.

Standart kelimelerde overlap başlangıcı kelime ortasına düşmez; overlap hedef miktardır ve yapı/kelime sınırında azalabilir. Chunk boyundan uzun boşluksuz öğeler ilerleme garantisi için hard-split edilir ve `hard_split=true` taşır. Whitespace-only parçalar gösterilmez; çok küçük bütçelerde kapsam kaybı ölçümde görünmelidir. Bu hafif ATX tarayıcı tam Markdown/HTML parser'ı veya PDF tablo çözücüsü değildir. Newline koruması semantik doğruluk kanıtı sayılmaz.

`experiments/eval/chunking_ablation_manifest.json` tek değişkeni ve aynı context bütçesini tanımlar. T08 qrel/grup metriği kaynakta geçerli olup kaybolan kanıtı paydadan çıkarmaz. Bu teslimde yalnız fixture karşılaştırması yapılır; gerçek model kalite/token/RSS gate'i `not_evaluated` kalır ve yeni strateji varsayılan yapılmaz.

`experiments/eval/` altında Türkçe bir değerlendirme seti bulunur: birbirine benzeyen iki belediye raporu ile bir izin yönergesi ve bunlar üzerine 41 soru. Sorular dört gruptadır: cevaplanabilir (Türkçe ve İngilizce), aynı konuda olup cevabı dokümanlarda bulunmayan "tuzak" sorular ve tamamen konu dışı sorular.

```bash
python experiments/evaluate.py --preflight-only  # kaynak/hash/span kontrolü; model yüklemez
python experiments/evaluate.py                  # gerçek retrieval; cache/bütçe gerekir
python experiments/evaluate.py --llm            # gerçek generation; ayrıca cache/bütçe gerekir
```

Betik uygulamayla aynı ayarları (environment ve `.env`) kullanır; `--threshold` ve `--top-k` ile geçici olarak değiştirilebilir. Ölçülenler:

- **Retrieval:** kaynak düzeyinde geçerli qrel OR/AND grubunun threshold öncesi hit@1, hit@k ve tamamlanma rank'ına göre group MRR değeri. Chunk kapsam kaybı paydadan çıkarılmaz; kaynak/oracle hatası ayrı kalite kapısını düşürür.
- **Threshold:** yalnız açık kalibrasyon verisinde `--tune` ve önceden tanımlı objective/constraint kullanılır. Eski 41 soru `legacy_regression` olarak korunur; yeni holdout değildir ve otomatik yeni eşik seçmez. Holdout'a kaydedilmiş seçim `--frozen-selection` ile sabit uygulanır.
- **Cevaplar (`--llm`):** `lexical_match`, ret ve hata; bağımsız kökenli semantik rubric ayrı tutulur. Lexical eşleşme doğru cevap garantisi değildir. Generation süresi T07 invocation bayrağına göre hesaplanır; model retleri ve başlatılmış hatalar da sayılır.

Her çalıştırma `experiments/results/` altına şema 2 JSON raporu yazar (git'e eklenmez); `--output` başka yerel hedef seçebilir. Dış dataset/corpus yolları desteklenir. Kaynak/model/prompt/index/split fingerprint'leri ve ham satırlar korunur; uyumsuz girdiler uyarısız birleştirilmez. Özel raporları public CI artifact'ına koymayın. Ayrıntılı şema, offset/context bütçesi ve ablation sözleşmesi: [experiments/eval/README.md](experiments/eval/README.md). Bu teslimde gerçek-model kalite/direnç baseline'ı ölçülmedi; `eval-history/fixture-baseline.json` yalnız sentetik mekanik örnektir.

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

- Index tek servis yazıcısı içindir; aynı `INDEX_DIR`'e ikinci servis OS lease alamaz. Read/reload nesneleri ayrı sözleşmedir.
- Her yükleme ve silmede candidate index ve tam NPZ yazımı O(N) iş yapar; bu donanımda kapasite/p95 ölçülmedi.
- PDF metin çıkarma, taranmış görüntüler için OCR yapmaz.
- Şu anda yalnızca PDF, TXT ve Markdown desteklenir.
- Cevap doğrulaması atıfları ve sayıları kontrol eder; sayı içermeyen iddiaların kanıta dayanıp dayanmadığını denetlemez.
- Değerlendirme seti küçüktür (41 soru, 3 doküman); sonuçlar yön gösterir, kesin değildir.
