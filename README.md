# 🧠 Cognitive Core Engine

**Cognitive Core**, statik makine öğrenmesi modellerinin kısıtlarını aşmak üzere geliştirilmiş; sistemdeki **"kör noktaları" (bilinmeyen / eksik sensör ve öznitelikleri)** anomali eşikleri ve histerezis kalkanıyla otonom tespit eden, dış dünyadan konsensüslü araştırma yürüten ve felaketsel unutma (catastrophic forgetting) yaşamaksızın **nörogenez (anlık mikro YSA modülü sentezi)** ile genişleyen otonom bir bilişsel mimaridir.

---

## 🎯 Bilişsel Döngüler (Cognitive Loops)

* 🟢 **LOOP_A (Refleks Çıkarımı):** Bilinen rejimlerde dış dünyaya sorgu atmadan, yerel bellek üzerinden mikrosaniyeler mertebesinde saf matris çıkarımı (Dış çağrı: 0).
* 🟡 **LOOP_B (Hafıza & Mühürleme):** Başarıyla doğrulanan durumların H3 uzman katmanına mühürlenmesi ve diske serileştirilmesi.
* 🟣 **LOOP_C (Dış Araştırma & Keşif):** Kör nokta anomalisi tespit edildiğinde araştırma ajanları üzerinden çapraz konsensüs ile eksik boyutun keşfedilmesi.
* 🔵 **LOOP_D (Nörogenez & Sentez):** Keşfedilen yeni eksen için çapa (anchor) testlerini geçecek şekilde sıfırdan mikro yapay sinir modülü sentezlenmesi.

---

## 📂 Dizin Mimarisi

```text
cognitive_core/
├── adapters/                  # Hexagonal Ports & Adapters (Dış adaptörler)
│   ├── research/              # Araştırma ajanı (Gemini / Mock konsensüs motoru)
│   └── storage/               # JSON tabanlı bellek deposu (json_memory_repo)
├── core/                      # Bağımsız iş mantığı ve çekirdek motor
│   ├── application/           # Servis orkestrasyonu ve portlar (ports, services)
│   ├── domain/                # Varlıklar, değer nesneleri ve etki alanı kuralları
│   ├── infrastructure/        # YSA motoru, modül yönlendirici, konsensüs motoru
│   └── interfaces/            # EngineBuilder ve Facade giriş noktaları
├── data/                      # Test verilerinin ekleneceği klasör (.csv, .txt, .tsv)
├── storage/                   # Kalıcı hafıza dizini (Çalışma anında oluşur)
│   └── knowledge_base.json    # Mühürlenmiş uzman modüller ve eksen uzayı
├── tests/                     # Birim, entegrasyon ve düşmanca (adversarial) testler
│   ├── adversarial/           # Sistem stres testleri
│   ├── integration/           # Uçtan uca otonom hat ve rejim testleri
│   └── unit/                  # Çekirdek birim testleri
├── run_engine.py              # Temel mimari ve kalıcılık doğrulama betiği
├── run_generic_demo.py        # Canlı sunum ve genel veri seti çalıştırıcısı
├── stress_test.py             # Düşmanca koşullar ve çoklu rejim stres testi
├── ARCHITECTURE.md            # Detaylı mimari şartname
└── MEMORY.md                  # Bellek ve mühürleme kuralları



🚀 Kurulum ve Gereksinimler
Proje saf Python (Pure Python) ile tasarlanmıştır; harici derin öğrenme kütüphanelerine (PyTorch, TensorFlow vb.) bağımlılığı yoktur.

Python Sürümü: Python 3.10 veya üzeri önerilir.

Bash
git clone <repo-url>
cd cognitive_core/cognitive_core



🧪 Kendi Veri Setinizle Canlı Demoyu Çalıştırma
Motor; CSV, TSV veya boşlukla ayrılmış metin (TXT) dosyalarını otomatik olarak tanır. Ayırıcıyı ve sütun başlıklarını kendi algılar. Son sütunu hedef (Y), önceki sütunları girdi öznitelikleri kabul eder.

1. Veri Dosyanızı Ekleyin
Test etmek istediğiniz veri dosyasını (örneğin ornek_veri.csv veya veriseti.txt) data/ klasörünün içine bırakın.

2. Demoyu Başlatın
Bash
# Otomatik kör nokta tespiti ile başlatma:
python run_generic_demo.py --data "data/ornek_veri.csv"
Parametreler (Opsiyonel):
--data: Veri dosyasının yolu (Varsayılan: data/train_FD001.txt)

--hide: Sisteme bilerek kör nokta tuzağı olarak saklamak istediğiniz sütun adı.

--steps: Kaç satırın işleneceği (Varsayılan: 150)

Örnek:

Bash
python run_generic_demo.py --data "data/ornek_veri.csv" --hide "sicaklik_sensoru" --steps 200


🖥️ Canlı Terminal Çıktısı Nasıl Okunur?
🟣 [DIŞ DÜNYADAN ARAŞTIRILDI]: Model başlangıçta kendisine verilmeyen ekseni (--hide ile gizlenen sütun) fark eder. Anomaliyi mevcut boyutlarla çözemeyince dış araştırma katmanını tetikler ve konsensüsle ekseni koordinat uzayına ekler.

🔵 [O ANDA ÖĞRENİLDİ / NÖROGENEZ]: Keşfedilen yeni eksen için milisaniyeler içinde sıfırdan bir mikro yapay sinir ağı modülü (m_syn_...) sentezlenir ve çapa testlerini geçerek H3 hafıza katmanına mühürlenir.

🟢 [ÖNCEDEN ÖĞRENİLEN / REFLEKS]: Sonraki adımların tamamında dış çağrı sayısı 0 olur. Model dış API'lere bağımlı kalmadan yerel CPU'dan mikrosaniyede çıkarım yapar.

🧠 Belleği Sıfırlama (Tabula Rasa)
Motor öğrendiği modülleri ve genişleyen eksen uzayını storage/knowledge_base.json dosyasına kaydeder. Yeni bir teste sıfır hafıza (Tabula Rasa) ile başlamak için hafıza dosyasını silmek yeterlidir:

Windows (PowerShell):

PowerShell
Remove-Item -Path "storage\knowledge_base.json" -ErrorAction SilentlyContinue


Linux / macOS (Bash):

Bash
rm -f storage/knowledge_base.json
🧩 Kod İçinden Yeni Eksen ve Motor Tanımlama
EngineBuilder akıcı arayüzü (fluent interface) üzerinden parametre uzayı, özel hiperparametreler ve araştırma ajanları kod içinden doğrudan yapılandırılabilir:

Python
from core.domain import DimensionAxis, GlobalParameterSpace
from core.interfaces import EngineBuilder
from core.infrastructure.research_agent import CrossConsensusEngine

# 1. Başlangıç Eksen Uzayını Tanımla
param_space = GlobalParameterSpace()
param_space.register_dimension(DimensionAxis(name="sensor_a", default_value=100.0, variance_threshold=0.1))
param_space.register_dimension(DimensionAxis(name="sensor_b", default_value=20.0, variance_threshold=0.05))

# 2. Bilişsel Çekirdeği İnşa Et
engine = (
    EngineBuilder()
    .with_parameter_space(param_space)
    .with_consensus_engine(CrossConsensusEngine(min_independent_sources=2))
    .with_hyperparameters(
        error_threshold=1.5,       # Anomali tetikleme eşiği
        anomaly_patience=3,        # Histerezis kalkanı (gürültü engelleme)
        consensus_threshold=0.70,  # Ajan konsensüs barajı
        regression_tolerance=10.0, # Çapa testi gerileme toleransı
    )
    .build()
)

# 3. Canlı Veri Akıt
telemetry = engine.step(
    raw_input={"sensor_a": 102.3, "sensor_b": 20.1},
    observed_outcome=15.4
)

print(f"Döngü: {telemetry['loop_executed']} | Aktif Modül: {telemetry.get('active_module_id')}")
🛡️ Otomatik Testleri Koşturma
Sistemdeki tüm değişmez kuralları (invariants) ve entegrasyon hatlarını doğrulamak için:

Bash
# Tüm birim ve entegrasyon testlerini çalıştır
python -m unittest discover -s tests

# Düşmanca rejim stres testini çalıştır
python stress_test.py