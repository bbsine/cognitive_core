# OTONOM BİLİŞSEL ÇEKİRDEK MOTORU (COGNITIVE CORE ENGINE) — MİMARİ ANAYASA

> **Doküman Statüsü:** Değiştirilemez Temel Referans (Sprint 0)  
> **Mimari Paradigma:** 5 Katmanlı, 3 Epistemik Havuzlu, Alan-Bağımsız (Domain-Agnostic) Clean Architecture  
> **Dağıtım Modeli:** Kapalı Kutu Çekirdek (Closed-Box Core Binary) & Açık Sözleşmeli Eklenti Ekosistemi (DIP/SPI)

---

## 1. SİSTEM VİZYONU VE TASARIM FELSEFESİ

**Bilişsel Çekirdek Motoru (Cognitive Core Engine)**; çalıştığı uygulama alanından (fizik, finans, robotik, siber güvenlik vb.) tamamen habersiz olan, dış dünyayı yalnızca soyut zaman-serisi tensörleri $X(t)$ ve gözlemlenen durum vektörleri $Y_{obs}$ üzerinden algılayan otonom bir epistemik çıkarım, teşhis ve topolojik büyüme motorudur.

Sistem; sabit (statik) ağırlıklı yapay sinir ağlarının aksine, **epistemik yanılabilirlik (fallibilism)** ilkesiyle çalışır:
1. Hiçbir kuralı veya modeli mutlak doğru ($P = 1.0$) kabul etmez.
2. Tahmin hatası ($\varepsilon$ aşımı) oluştuğunda mevcut bilgisini karantinaya alır ($H_2$).
3. Hatayı önce içsel parametre uzayında ($\Omega_{global}$) arar; açıklayamazsa karşı-olgusal (counterfactual) teşhisle **eksik boyutu** ($\theta_{aday}$) izole eder.
4. Dış kaynaklardan çapraz konsensüsle doğruladığı yeni boyutu ($\theta_{kesif}$) global uzaya ekler, bu boyut için yeni bir mikro sinir ağı ($m_{yeni}$) sentezler ve geriye dönük regresyon testini (Catastrophic Forgetting Koruması) geçtikten sonra aktif çekirdeğe ($H_3$) topolojik olarak birleştirir.

---

## 2. REFERANS MİMARİ VE AKIŞ ŞEMASI

Aşağıdaki akış şeması sistemin değiştirilemez omurgasıdır. Tüm katmanlar, portlar ve iş akışları (Use Cases) bu topolojiye birebir uymak zorundadır:

```text
========================================================================================================================
                                       REFERANS MİMARİ VE AKIŞ ŞEMASI
========================================================================================================================
  [Dış Gözlem / Çekirdek Girdi: X(t)]
                     │
                     ▼
  +──────────────────────────────────────────────────────────────────────────+
  │                        1. HAVUZ: HAM VERİ GİRİŞİ (H_1)                   │
  +──────────────────────────────────────────────────────────────────────────+
                     │
                     ▼
  [Nitelik Ayrıştırma Operatörü: Φ(X)]
  - Girdiyi bileşen tensörlerine ayrıştır: V = {v_1, v_2, ..., v_n}
  - İnvaryans Analizi: Çıktı üzerinde etkisi olmayan sabit/bağlamsız değişkenleri maskele
                     │
                     ▼
  [Seyrek Dikkat Kapısı: G_sparse(V)]
  - Yalnızca anlık değişim gösteren (Δv_i != 0) veya durum geçişi tetikleyen değişkenleri seç
  - Aktif çalışma alt-uzayını filtrele: V_aktif ⊆ V (|V_aktif| << |V|)
                     │
                     │  V_aktif (Dinamik Çalışma Vektörü)
                     ▼
========================================================================================================================
                              KATMAN 2 & 3: KADEMELİ İÇ SORGULAMA VE ÖNGÖRÜ MOTORU
========================================================================================================================
  +──────────────────────────────────────────────────────────────────────────+
  │                3. HAVUZ: AKTİF ÇEKİRDEK (DOĞRU KABUL EDİLENLER - H_3)    │
  │  - Router Ağ (Yönlendirici Topoloji)                                     │
  │  - Aktif Uzman Ağlar: M = {m_1, m_2, ..., m_k}                           │
  │  - Epistemik Güven Katsayısı: P(kural) ∈ [P_min, P_max), P < 1.0         │
  +──────────────────────────────────────────────────────────────────────────+
                     │
                     ▼
  [DÖNGÜ A - SEVİYE 1: Doğrudan Model Tahmini]
  - Model Tahmini: Y_pred = M(V_aktif)
  - Gözlemlenen Durum: Y_obs
                     │
        ┌────────────┴────────────┐
        │                         │
  [|Y_obs - Y_pred| <= ε]   [|Y_obs - Y_pred| > ε] (ANOMALİ / ÇELİŞKİ)
        │                         │
        ▼                         ▼
  (DÖNGÜ A: Pekiştirme)     +──────────────────────────────────────────────────+
  P += η * (P_max - P)      │ İlgili Modülün Güvenini Düşür: P = P * γ (γ < 1) │
  (Asimptotik Artış)        │ Durumu ve Kuralı 2. HAVUZ'a (H_2 Karantina) Taşı │
                            +──────────────────────────────────────────────────+
                                          │
                                          ▼
                            [DÖNGÜ B - SEVİYE 2: İç Bellek & Graf Taraması]
                            - Global parametre uzayında (Ω_global) mevcut olup
                              bu bağlamda aktifleşmemiş bir boyut var mı?
                                          │
                                  ┌───────┴───────┐
                                  │               │
                        [İç Uzayda Açıklandı]   [İç Uzay Yetersiz (Boyut Eksik)]
                                  │               │
                                  ▼               ▼
                        (H_2'den Doğrula,       [SEVİYE 3: Karşı-Olgusal Teşhis]
                         Alt-Kuralı Derle)      - Durum Değişim İzolasyonu: Δ(Durum)
                                                - Eksik Boyut Hipotezi: θ_aday
                                                - Hedef Sorgu: Q(Δ, V_aktif, Y_obs)
                                                        │
                                                        ▼
========================================================================================================================
                                  KATMAN 4: ÇAPRAZ DIŞ ARAŞTIRMA & BOYUT GENİŞLETME
========================================================================================================================
                                                        │
                                                        ▼
  [Hedef Odaklı Dış Bilgi Çekimi: Q]
  - Harici Veri Kaynakları (Metin / API / Veritabanı)
                     │
                     ▼
  [DÖNGÜ C: Çapraz Konsensüs & Zehirlenme Koruması]
  - Bağımsız N kaynak üzerinden semantik / istatistiksel doğrulama
  - Konsensüs Denetimi: C(Kaynaklar) >= Eşik
        ├─ [Tutarsız / Yetersiz] ──> H_2'de dondur, beklet
        └─ [Konsensüs Başarılı]  ──> Yeni Parametre/Boyut Onaylandı: θ_kesif
                     │
                     ▼
  +──────────────────────────────────────────────────────────────────────────+
  │                  GLOBAL PARAMETRE UZAYI GENİŞLEMESİ (Ω_global)           │
  │  - Ω_global = Ω_global ∪ {θ_kesif}                                       │
  │  - Yeni boyut tüm varlık uzayı için potansiyel eksen haline gelir        │
  +──────────────────────────────────────────────────────────────────────────+
                     │
                     ▼
========================================================================================================================
                            KATMAN 5: OTONOM SİNİR AĞI DERLEME & TOPOLOJİK BÜYÜME
========================================================================================================================
                     │
                     ▼
  [Yeni Mikro Alt-Ağ Şablonu (Neurogenesis): m_yeni]
  - Girdi Boyutu: Dim(V_aktif ∪ {θ_kesif})
  - Çıktı: Hedef Fonksiyon Eşlemesi
                     │
                     ▼
  [İç Simülasyonda Hızlı Ön-Eğitim]
  - Sınır şartları ve sentetik varyasyonlarla yerel optimizasyon
                     │
                     ▼
  +──────────────────────────────────────────────────────────────────────────+
  │              2. HAVUZ: GÖLGE DAL (SHADOW BRANCH) TEST ALANI (H_2)        │
  +──────────────────────────────────────────────────────────────────────────+
                     │
                     ▼
  [DÖNGÜ D: Regresyon ve Stabilite Testi (Geriye Dönük Koruma)]
  - m_yeni geçmişte doğrulanmış çapa verilerini bozuyor mu?
  - Loss_eski <= Kabul_Toleransı
                     │
        ┌────────────┴────────────┐
        │                         │
  [Bozunum Var]             [Bozunum Yok (Stabilite Onaylandı)]
        │                         │
        ▼                         ▼
  (DÖNGÜ D: Red)            [Topolojik Birleştirme (Architecture Merge)]
  H_2'de Karantinada Tut,   1. Router Ağa dinamik geçiş kapısı ekle
  Sınır Şartını Daralt      2. m_yeni modülünü H_3'e kalıcı olarak bağla
                            3. Kuralı H_3'e "Doğru Kabul Edilen" olarak terfi ettir
                                  │
                                  ▼
========================================================================================================================
```

---

## 3. EPİSTEMİK HAVUZLAR VE 5 KATMANLI BİLİŞSEL HİYERARŞİ

### 3.1. Üç Epistemik Havuz ($H_1, H_2, H_3$)
Sistem, bilginin doğrulanmışlık derecesine göre veriyi ve modelleri üç izole bellek havuzunda yönetir:

1. **$H_1$ — Ham Veri Giriş Havuzu (Raw Sensory & Input Buffer):**
   * Dış dünyadan akan işlenmemiş, yüksek boyutlu ve gürültülü gözlem tensörlerinin $X(t)$ ilk temas noktasıdır.
   * Epistemik değeri sıfırdır; burada hiçbir kural çıkarımı yapılmaz, yalnızca zamansal pencereleme ve ayrıştırma öncesi tamponlama uygulanır.
2. **$H_2$ — Karantina ve Gölge Dal Havuzu (Quarantine & Shadow Branch Pool):**
   * Sistemin **epistemik laboratuvarı ve izolasyon alanıdır**. Üç tür varlık barındırır:
     1. $H_3$'te iken tahmin hatası ($\Vert{}Y_{obs} - Y_{pred}\Vert{} > \varepsilon$) ürettiği için güveni düşürülerek ($\gamma$) incelemeye alınan anomalili durumlar ve kurallar.
     2. Dış araştırmada (Döngü C) yeterli konsensüs sağlayamamış ($C < \text{Eşik}$) dondurulmuş hipotezler.
     3. Yeni keşfedilen boyutlarla ($\theta_{kesif}$) sentezlenen ve $H_3$'e terfi etmeden önce iç simülasyon ile geriye dönük regresyon testine (Döngü D) tabi tutulan aday mikro ağlar ($m_{yeni}$).
3. **$H_3$ — Aktif Çekirdek Havuzu (Active Epistemic Core - "Doğru Kabul Edilenler"):**
   * Yalnızca regresyon ve tutarlılık testlerini geçmiş, aktif olarak çıkarım yapan yönlendirici ağı (Router Network) ve uzman mikro ağları $M = \{m_1, m_2, \dots, m_k\}$ barındırır.
   * Buradaki her kural/modül yanlışlanabilir (falsifiable) niteliktedir ve $P \in [P_{min}, P_{max})$ aralığında dinamik bir güven katsayısına sahiptir.

### 3.2. Beş Bilişsel Katman
* **Katman 1 (Algısal Ayrıştırma ve Seyrek Dikkat):** $H_1$ üzerinden gelen $X(t)$ tensörünü $\Phi(X)$ ile bileşenlerine ayırır, invaryans analiziyle etkisiz boyutları maskeler ve $G_{sparse}(V)$ kapısıyla yalnızca aktif alt-uzayı ($V_{aktif}$) seçer.
* **Katman 2 (Doğrudan Öngörü ve Epistemik Değerlendirme - Döngü A):** $V_{aktif}$ vektörünü $H_3$ içindeki Router üzerinden ilgili uzman ağa ($m_i$) yönlendirir, $Y_{pred}$ üretir ve $Y_{obs}$ ile karşılaştırarak pekiştirme veya karantina kararını verir.
* **Katman 3 (Kademeli İç Sorgulama ve Karşı-Olgusal Teşhis - Döngü B):** Karantinaya alınan anomalinin mevcut global parametre uzayındaki ($\Omega_{global}$) pasif boyutlarla açıklanıp açıklanamayacağını denetler (Seviye 2). Açıklanamıyorsa karşı-olgusal analizle eksik boyut hipotezi ($\theta_{aday}$) ve hedef sorgu $Q(\Delta, V_{aktif}, Y_{obs})$ üretir (Seviye 3).
* **Katman 4 (Çapraz Dış Araştırma ve Boyut Genişletme - Döngü C):** $Q$ sorgusunu dış bilgi kaynaklarına iletir, bağımsız $N$ kaynak üzerinden zehirlenme korumalı konsensüs denetimi yapar ve onaylanan boyutu ($\theta_{kesif}$) $\Omega_{global}$ uzayına katar.
* **Katman 5 (Otonom Sinir Ağı Derleme ve Topolojik Büyüme - Döngü D):** $\text{Dim}(V_{aktif} \cup \{\theta_{kesif}\})$ girdili yeni bir mikro alt-ağ ($m_{yeni}$) oluşturur, $H_2$ gölge dalında ön-eğitim ve geçmiş çapa verileri (anchor data) üzerinde regresyon testi uygular; bozunum yoksa Router ağını güncelleyerek modülü $H_3$'e birleştirir.

---

## 4. MATEMATİKSEL VE EPİSTEMİK KURALLAR (FORMALİZM)

### 4.1. Nitelik Ayrıştırma ($\Phi$) ve İnvaryans Maskeleme
Ham girdi $X(t)$, ayrıştırma operatörü $\Phi$ ile $n$ boyutlu bileşen vektörüne dönüştürülür:
$$\Phi(X(t)) \mapsto V(t) = \{v_1(t), v_2(t), \dots, v_n(t)\}$$

**İnvaryans Analizi:** Belirli bir $\Delta t$ gözlem penceresi boyunca hedef durum değişimlerine karşı kısmi türevi (duyarlılığı) veya karşılıklı bilgisi (Mutual Information - $I$) sıfıra yakınsayan sabit/bağlamsız bileşenler maskelenir:
$$M_{inv}(v_i) = \mathbb{I}\left( I(v_i; Y_{obs}) > \tau_{inv} \right)$$

### 4.2. Seyrek Dikkat Kapısı ($G_{sparse}$) ve Eşik Denklemleri
İnvaryans filtresinden geçen bileşenler, yalnızca anlık değişim gösterdiklerinde veya bir durum geçişini (state transition) tetiklediklerinde aktif çalışma alt-uzayına ($V_{aktif}$) kabul edilir:
$$G_{sparse}(V) = \left\{ v_i \in V \;\middle\vert{}\; \vert{}\Delta v_i(t)\vert{} > \theta_{\Delta}^{(i)} \;\lor\; \mathcal{S}_{trig}(v_i(t)) = 1 \right\} = V_{aktif}$$

Burada:
* $\Delta v_i(t) = v_i(t) - v_i(t-1)$ zamansal fark operatörüdür.
* $\theta_{\Delta}^{(i)} > 0$, $i$-inci boyut için mikro-gürültü bastırma eşiğidir (noise floor threshold).
* $\mathcal{S}_{trig}(v_i(t)) \in \{0, 1\}$, değişkenin kritik bir sınır değerini (boundary regime) aştığını belirten durum geçiş tetikleyicisidir.
* **Seyreklik Şartı:** Sistem her zaman $V_{aktif} \subseteq V$ ve $\vert{}V_{aktif}\vert{} \ll \vert{}V\vert{}$ kısıtını koruyarak hesaplama karmaşıklığını aktif alt-uzaya indirger.

### 4.3. Epistemik Güven Sınırları ($P < 1.0$ İnvaryantı)
$H_3$ içindeki hiçbir kural veya uzman ağ $m_k$ mutlak kesinliğe ulaşamaz. Güven katsayısı $P$ yarı-açık bir aralıkta tanımlıdır:
$$P(\text{kural}) \in [P_{min}, P_{max}), \quad 0 < P_{min} < P_{max} < 1.0$$

* **Üst Sınır ($P_{max} = 1.0 - \delta$, örn. $0.9999$):** $P \ge 1.0$ olması matematiksel olarak yasaktır. Eğer $P = 1.0$ olursa, sistem o kuralı dogmalaştırır ve gelecekteki rejim kaymalarına (concept drift / black swan) karşı körleşir.
* **Alt Sınır ($P_{min}$):** Güveni bu eşiğin altına düşen modül $H_3$'te kalamaz; derhal $H_2$ karantinasına 강등 (demote) edilir.

### 4.4. Döngü A: Asimptotik Güven Artışı ve Ceza (Decay) Dinamikleri
Tahmin hatası $e(t) = \Vert{}Y_{obs}(t) - Y_{pred}(t)\Vert{}_p$ ve kabul edilebilir hata toleransı $\varepsilon > 0$ için:

1. **Başarılı Tahmin ($e(t) \le \varepsilon$ — Pekiştirme):**
   Güven katsayısı doğrusal değil, $P_{max}$ sınırına doğru **asimptotik** olarak artar:
   $$P_{t+1} = P_t + \eta \cdot (P_{max} - P_t), \quad \eta \in (0, 1)$$
   *Özellik:* $P_t \to P_{max}$ iken $(P_{max} - P_t) \to 0$ olduğundan artış hızı doğal olarak yavaşlar ve $P_{t+1}$ hiçbir sonlu adımda $P_{max}$ değerini aşamaz.

2. **Anomali / Çelişki ($e(t) > \varepsilon$ — Güven Düşürme ve Karantina):**
   İlgili modülün güveni geometrik bir ceza çarpanı ($\gamma$) ile düşürülür:
   $$P_{t+1} = P_t \cdot \gamma, \quad \gamma \in (0, 1)$$
   Anomaliyi üreten durum vektörü $(V_{aktif}, Y_{obs}, Y_{pred})$ ve ilgili kural bağlamı derhal $H_2$ (Karantina) havuzuna taşınır ve Döngü B tetiklenir.

### 4.5. Döngü B: İç Uzay Taraması ve Karşı-Olgusal Teşhis
* **Seviye 2 (İç Bellek & Graf Taraması):** Global parametre uzayında bulunan fakat o anki $V_{aktif}$ içinde yer almayan pasif boyutlar kümesi $\Omega_{pasif} = \Omega_{global} \setminus V_{aktif}$ taranır. Eğer $\exists \theta_j \in \Omega_{pasif}$ eklendiğinde hata $\Vert{}Y_{obs} - M(V_{aktif} \cup \{\theta_j\})\Vert{} \le \varepsilon$ düzeyine iniyorsa, anomali iç uzayda açıklanmış kabul edilir ve $H_2$'de alt-kural derlenir.
* **Seviye 3 (Karşı-Olgusal Teşhis):** $\Omega_{global}$ içindeki hiçbir mevcut boyut hatayı açıklayamıyorsa, sistem durum değişim vektörünü izole eder:
  $$\Delta(\text{Durum}) = \mathcal{F}_{diff}(Y_{obs}, Y_{pred} \mid V_{aktif})$$
  Bu farkı sıfırlayacak en küçük teorik gizli değişken (latent variable) **Eksik Boyut Hipotezi** ($\theta_{aday}$) olarak tanımlanır ve yapılandırılmış bir dış sorgu descriptor'ına dönüştürülür: $Q(\Delta, V_{aktif}, Y_{obs})$.

### 4.6. Döngü C: Çapraz Konsensüs ve Zehirlenme Koruması (Poisoning Protection)
$Q$ sorgusu üzerinden bağımsız $N$ harici kaynaktan ($S = \{s_1, s_2, \dots, s_N\}$, $N \ge N_{min}$) aday boyut verileri çekilir. Veri zehirlenmesini (data poisoning) ve halüsinasyonu önlemek için ağırlıklı çapraz konsensüs fonksiyonu çalıştırılır:
$$C(S) = \frac{1}{\sum_{i=1}^N w_i} \sum_{i=1}^N w_i \cdot \text{Sim}_{sem/stat}(s_i, \bar{s}_{robust})$$
* Burada $\bar{s}_{robust}$, aykırı değerleri (Byzantine outliers) budanmış medyan/temsilci dağılımı; $w_i$ ise kaynağın bağımsızlık ve tarihsel güven ağırlığını temsil eder.
* Eğer $C(S) \ge \tau_{konsensus}$ ise $\theta_{aday} \to \theta_{kesif}$ olarak onaylanır ve global uzay genişletilir:
  $$\Omega_{global} \leftarrow \Omega_{global} \cup \{\theta_{kesif}\}$$
* Eğer $C(S) < \tau_{konsensus}$ ise hipotez reddedilmez ancak $H_2$'de **dondurulmuş (frozen)** statüde bekletilir.

### 4.7. Döngü D: Neurogenesis, Regresyon Koruması ve Topolojik Birleştirme
Onaylanan $\theta_{kesif}$ ile genişletilmiş girdi uzayı $\text{Dim}(V_{aktif} \cup \{\theta_{kesif}\})$ üzerinde yeni bir mikro alt-ağ ($m_{yeni}$) ilklendirilir ve $H_2$ gölge dalında sentetik sınır şartlarıyla ön-eğitime tabi tutulur.

**Geriye Dönük Koruma (Catastrophic Forgetting / Regression Test):**
$m_{yeni}$ modülünün ve güncellenmiş Router topolojisinin, geçmişte doğrulanmış çapa veri kümesi ($\mathcal{D}_{anchor} \subset H_3$) üzerindeki kaybı ölçülür:
$$\text{Loss}_{eski} = \frac{1}{\vert{}\mathcal{D}_{anchor}\vert{}} \sum_{(x, y) \in \mathcal{D}_{anchor}} \mathcal{L}\big(M_{aday}(x), y\big)$$

* **Koşul:** $\text{Loss}_{eski} \le \tau_{kabul}$ (Kabul Toleransı).
* Eğer bozunum varsa ($\text{Loss}_{eski} > \tau_{kabul}$), birleştirme reddedilir; $m_{yeni}$ modülü $H_2$'de tutulur ve aktivasyon sınır şartları (router gating sınırları) daraltılarak yeniden izole edilir.
* Eğer bozunum yoksa, **Topolojik Birleştirme (Architecture Merge)** gerçekleşir: Router ağa yeni geçiş kapısı eklenir, $m_{yeni} \in H_3$ olarak bağlanır ve başlangıç güven katsayısı $P_0 \in [P_{min}, P_{max})$ atanır.

---

## 5. CLEAN ARCHITECTURE KATMAN SINIRLARI

Sistem, bağımlılıkların yalnızca dıştan içe doğru aktığı (**The Dependency Rule**) 4 katmanlı Clean Architecture yapısında inşa edilir:

```text
+-----------------------------------------------------------------------+
|  4. INTERFACES (İstemci API / SDK Facade / C-ABI / CLI)               |
|     +-----------------------------------------------------------+     |
|     |  3. INFRASTRUCTURE (Bellek, Dış Ajan ve Hesaplama Adapt.) |     |
|     |     +-----------------------------------------------+     |     |
|     |     |  2. APPLICATION (Use Cases & SPI Portları)    |     |     |
|     |     |     +-----------------------------------+     |     |     |
|     |     |     |  1. DOMAIN (Saf Epistemik Mantık) |     |     |     |
|     |     |     +-----------------------------------+     |     |     |
|     |     +-----------------------------------------------+     |     |
|     +-----------------------------------------------------------+     |
+-----------------------------------------------------------------------+
         Bağımlılık Yönü: Dış Katmanlar ──────► İç Katmanlar
```

### 5.1. Domain Katmanı (`src/core/domain/`)
* **Rolü:** Bilişsel çekirdeğin kalbidir. Tüm epistemik kuralları, matematiksel invaryantları, havuz geçiş şartlarını ve varlık tanımlarını barındırır.
* **İçerik:**
  * *Entities:* `EpistemicRule`, `SpecialistModuleMetadata`, `RouterTopologyGraph`, `QuarantineRecord`, `GlobalParameterSpace`.
  * *Value Objects:* `ConfidenceScore` ($P < 1.0$ zorlayan), `TensorVector`, `DimensionDescriptor` ($\theta$), `AnomalyDelta`, `TargetQuery` ($Q$), `ConsensusMetric`.
  * *Domain Services / Operators:* `FeatureDecomposer` ($\Phi$), `SparseAttentionGate` ($G_{sparse}$), `EpistemicReinforcer`, `CounterfactualDiagnoser`.
  * *Domain Events & Exceptions:* `AnomalyDetectedEvent`, `DimensionDiscoveredEvent`, `EpistemicInvariantViolationException`, `RegressionFailureException`.
* **Sınır Kuralları:**
  * **Sıfır Harici Kütüphane Bağımlılığı:** Yalnızca dilin standart kütüphanesi (STL / Python Standard Library) kullanılabilir. PyTorch, TensorFlow, NumPy, SQLite, HTTP istemcileri veya işletim sistemi soketleri bu katmana **asla** giremez.
  * **Mutlak Alan Bağımsızlığı:** Kod içinde `price`, `temperature`, `velocity`, `stock` gibi alana özgü hiçbir kavram bulunamaz.

### 5.2. Application Katmanı (`src/core/application/`)
* **Rolü:** Domain nesnelerini kullanarak sistemin 5 katmanlı bilişsel döngülerini (Döngü A, B, C, D) orkestre eder ve dış dünyayla iletişim için soyut **Port (SPI - Service Provider Interface)** sözleşmelerini tanımlar.
* **İçerik:**
  * *Use Cases (Interactors):*
    * `ProcessObservationUseCase` (Katman 1 & Döngü A)
    * `InterrogateInternalSpaceUseCase` (Döngü B - Seviye 2 & 3)
    * `ExecuteExternalResearchUseCase` (Katman 4 & Döngü C)
    * `CompileAndMergeSubnetworkUseCase` (Katman 5 & Döngü D)
  * *Output Ports (SPI):*
    * `IMemoryRepository`: $H_1, H_2, H_3$, $\Omega_{global}$ ve $\mathcal{D}_{anchor}$ verilerinin kalıcılık/sorgulama arayüzü.
    * `IResearchAgent`: $Q(\Delta, V_{aktif}, Y_{obs})$ sorgusunu dış kaynaklarda koşturup ham kanıtları getiren ajan arayüzü.
    * `INeuralEnginePort`: Soyut topoloji tanımlarını gerçek tensör/sinir ağı hesaplamalarına (Forward pass, Pre-training, Loss hesabı) dönüştüren hesaplama motoru arayüzü.
    * `IConsensusVerifierPort`: $N$ kaynaktan gelen yanıtların semantik/istatistiksel konsensüs skorunu hesaplayan arayüz.
* **Sınır Kuralları:** Yalnızca `Domain` katmanına bağımlıdır. Veritabanı teknolojisini, hangi LLM/API'nin kullanıldığını veya sinir ağının CPU/GPU üzerinde nasıl koştuğunu bilmez.

### 5.3. Infrastructure Katmanı (`src/adapters/infrastructure/`)
* **Rolü:** Application katmanındaki SPI portlarının somut (concrete) implementasyonlarını içerir.
* **İçerik:**
  * *Memory Adapters:* `InMemoryGraphRepository`, `SqliteEpistemicRepository`, `VectorAnchorStore`.
  * *Research Agent Adapters:* `MultiSourceApiResearchAgent`, `LocalCorpusResearchAgent`, `MockResearchAgent`.
  * *Neural Compute Adapters:* `CppEigenNeuralAdapter`, `LibTorchComputeAdapter`, `PyTorchSubnetworkAdapter`.
* **Sınır Kuralları:** Üçüncü parti tüm kütüphaneler ve I/O işlemleri bu katmanda kapsüllenir.

### 5.4. Interfaces / SDK Katmanı (`src/interfaces/sdk/`)
* **Rolü:** Dış sistemlerin Bilişsel Çekirdek Motoru'nu bir kütüphane veya servis olarak tüketmesini sağlayan giriş noktasıdır.
* **İçerik:** `CognitiveCoreFacade`, `EngineBuilder` (DI Container kurulumu), C-ABI / Python bağlamaları (Bindings), telemetri dinleyicileri.

---

## 6. FİKRİ MÜLKİYET (IP) KORUMA VE İKİLİ DAĞITIM MİMARİSİ

Sistemin bilişsel orkestrasyon algoritmaları, karşı-olgusal teşhis mantığı ve topolojik büyüme kuralları projenin **çekirdek fikri mülkiyetidir (Core IP)**. Bu mülkiyeti korumak için aşağıdaki izolasyon mimarisi uygulanır:

```text
+=======================================================================+
|                KAPALI KUTU ÇEKİRDEK (CLOSED-BOX BINARY)               |
|  [ Domain Katmanı ] + [ Application Katmanı (Use Cases) ]             |
|  -> Derlenmiş C++ Shared Library (.so / .dll) veya Cython (.pyd)      |
|  -> Sembolleri temizlenmiş (stripped), iç algoritmaları gizli         |
+===================================+===================================+
                                    |
          Açık Sözleşmeler (Public Headers / Abstract SPI Ports)
          - IMemoryRepository
          - IResearchAgent
          - INeuralEnginePort
          - CognitiveCoreFacade (Giriş/Çıkış SDK'sı)
                                    | (Dependency Inversion - DIP)
                                    ▼
+-----------------------------------------------------------------------+
|             AÇIK EKLENTİ EKOSİSTEMİ (OPEN PLUG-IN ADAPTERS)           |
|  - Özel Veritabanı Adaptörleri (PostgreSQL, Redis, Milvus vb.)        |
|  - Özel Dış Araştırma Ajanları (Web API, Finans/Fizik Veri Sağlayıcı) |
|  - Özel Donanım Hızlandırıcılar (CUDA, Metal, ONNX Runtime)           |
+-----------------------------------------------------------------------+
```

1. **Derlenebilir Çekirdek İzolasyonu:** `Domain` ve `Application` katmanları hiçbir dış kütüphaneye bağımlı olmadığı için bağımsız olarak derlenip kapalı kaynaklı bir ikili paket (`.so`, `.dll`, `.dylib` veya `.pyd`) haline getirilir.
2. **Açık Sözleşmeli Genişletilebilirlik (Open Contracts via DIP):** Dış geliştiricilere veya entegratör sistemlere yalnızca `Application` katmanının soyut port tanımları (Public Header / Abstract Base Classes) ve `SDK Facade` arayüzü açılır.
3. **Çalışma Zamanı Enjeksiyonu (Runtime Dependency Injection):** İstemci uygulama, kendi alanına özgü veri kaynaklarını (`IResearchAgent`) ve bellek altyapısını (`IMemoryRepository`) açık sözleşmelere göre kodlar ve çalışma zamanında kapalı çekirdeğe enjekte eder. Çekirdek, enjekte edilen adaptörleri yalnızca kendi epistemik kuralları çerçevesinde yönetir.