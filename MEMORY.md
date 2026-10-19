# PROJE HAFIZASI VE MİMARİ DURUM KAYDI (`MEMORY.md`)

> **Proje Adı:** Otonom Bilişsel Çekirdek Motoru (Cognitive Core Engine)  
> **Mevcut Durum:** **Sprint 0 — Mimari Anayasa ve Taslak Belirlendi**  
> **Aktif Kodlama Durumu:** Henüz uygulama kodu yazılmadı (Sıfır Kod Aşaması / Mimari Dondurma).

---

## 1. MEVCUT DURUM ÖZETİ (SPRINT 0)

* **Tamamlanan Çıktılar:**
  * `ARCHITECTURE.md`: 5 katmanlı ve 3 epistemik havuzlu ($H_1, H_2, H_3$) bilişsel akış şeması, matematiksel formalizm ($\Phi, G_{sparse}$, Döngü A/B/C/D denklemleri), Clean Architecture katman sınırları ve Kapalı Kutu (Closed-Box) IP koruma mimarisi tanımlandı ve donduruldu.
  * `MEMORY.md`: Sprint yol haritası, değiştirilemez sistem invaryantları ve mimari karar kayıtları oluşturuldu.
* **Bir Sonraki Adım:** Sprint 1'e geçiş onayı ile birlikte sıfır bağımlılıklı `Domain` katmanının inşası.

---

## 2. SPRINT YOL HARİTASI (ROADMAP)

Proje, içten dışa doğru (Inside-Out Clean Architecture) 5 ardışık Sprint halinde geliştirilecektir. Bir önceki sprintin birim ve invaryant testleri %100 geçmeden bir sonraki sprinte geçilemez.

### 🏁 Sprint 0: Mimari Anayasa ve Taslak (TAMAMLANDI)
* Referans ASCII mimari şemasının sabitlenmesi.
* Epistemik kuralların ve matematiksel sınırların dokümantasyonu (`ARCHITECTURE.md`).
* Proje hafızası ve invaryant kurallarının kaydı (`MEMORY.md`).

### ⏳ Sprint 1: Domain Katmanı (Entities, Value Objects, Events, Exceptions)
* **Hedef:** Sıfır harici kütüphane bağımlılığıyla saf epistemik ve matematiksel çekirdeğin kodlanması.
* **Teslimatlar:**
  * *Value Objects:* `ConfidenceScore` ($P \in [P_{min}, P_{max}), P < 1.0$ doğrulamalı), `ComponentVector` ($V$), `ActiveSubspaceVector` ($V_{aktif}$), `DimensionAxis` ($\theta$), `TargetQuery` ($Q$).
  * *Entities & Aggregates:* `EpistemicPool` ($H_1, H_2, H_3$ durum makineleri), `SpecialistModule` ($m_k$), `RouterTopology`, `GlobalParameterSpace` ($\Omega_{global}$), `QuarantineContext`.
  * *Domain Operators:* $\Phi(X)$ ayrıştırma ve invaryans maskeleme mantığı, $G_{sparse}(V)$ seyrek dikkat kapısı, Döngü A asimptotik güven güncelleme ve $\gamma$ ceza fonksiyonları.
  * *Domain Events & Exceptions:* `AnomalyQuarantinedEvent`, `DimensionHypothesisCreatedEvent`, `InvariantViolationException`.

### ⏳ Sprint 2: Application Katmanı & Portlar (SPI Sözleşmeleri)
* **Hedef:** Çekirdeğin dış dünya ile konuşacağı soyut sözleşmelerin (Dependency Inversion) ve veri transfer nesnelerinin (DTO) tanımlanması.
* **Teslimatlar:**
  * `IMemoryRepository`: $H_1, H_2, H_3$ havuzları, $\Omega_{global}$ parametre uzayı ve çapa verileri ($\mathcal{D}_{anchor}$) için CRUD ve graf sorgu sözleşmesi.
  * `IResearchAgent`: Karşı-olgusal teşhis çıktısı olan $Q(\Delta, V_{aktif}, Y_{obs})$ sorgusunu dış kaynaklara iletecek otonom araştırma ajanı sözleşmesi.
  * `INeuralEnginePort`: Mikro alt-ağ ($m_{yeni}$) ilklendirme, iç simülasyon ön-eğitimi, ileri besleme (inference) ve regresyon kaybı ($\text{Loss}_{eski}$) hesaplama sözleşmesi.
  * `IConsensusEvaluator`: $N$ bağımsız kaynaktan dönen dış verilerin zehirlenme (poisoning) ve konsensüs ($C \ge \text{Eşik}$) doğrulama sözleşmesi.

### ⏳ Sprint 3: Use Case'ler & Bilişsel Döngü Orkestrasyonu
* **Hedef:** 5 katmanlı bilişsel akışın ve A, B, C, D döngülerinin uçtan uca orkestrasyonu.
* **Teslimatlar:**
  * `LoopA_DirectPredictionUseCase`: $H_1 \to \Phi \to G_{sparse} \to H_3$ tahmini, pekiştirme veya $H_2$'ye karantina aktarımı.
  * `LoopB_InternalInterrogationUseCase`: $\Omega_{global}$ pasif boyut taraması (Seviye 2) ve başarısızlık durumunda karşı-olgusal teşhis ile $\theta_{aday}$ ve $Q$ üretimi (Seviye 3).
  * `LoopC_ExternalResearchAndExpansionUseCase`: `IResearchAgent` üzerinden çoklu kaynak sorgulama, konsensüs denetimi ve başarılı ise $\Omega_{global} \cup \{\theta_{kesif}\}$ genişletmesi; başarısız ise $H_2$'de dondurma.
  * `LoopD_NeurogenesisAndMergeUseCase`: $m_{yeni}$ derleme, iç simülasyon ön-eğitimi, $H_2$'de geriye dönük regresyon testi ($\text{Loss}_{eski} \le \text{Kabul\_Toleransı}$) ve test geçilirse $H_3$ Router topolojisine kalıcı birleştirme (Architecture Merge).

### ⏳ Sprint 4: Infrastructure & Plug-in Adaptörleri
* **Hedef:** Application portlarının somut altyapı bileşenleriyle gerçeklenmesi.
* **Teslimatlar:**
  * Bellek Adaptörleri (In-Memory / Kalıcı Graf ve Vektör Havuzu implementasyonları).
  * Dış Araştırma Ajanı Adaptörleri (Çoklu API / Veritabanı / Metin madenciliği bağlayıcıları ve Konsensüs doğrulayıcıları).
  * Sinir Ağı Hesaplama Adaptörleri (Mikro alt-ağların dinamik oluşturulması ve eğitilmesi için C++/Python hesaplama motoru adaptörleri).

### ⏳ Sprint 5: SDK Facade & İstemci Dağıtım Arayüzü
* **Hedef:** Çekirdeğin kapalı kutu (Closed-Box Binary) olarak paketlenmesi ve dış sistemler için temiz bir SDK arayüzü sunulması.
* **Teslimatlar:**
  * `CognitiveCoreSDK` / `EngineFacade`: Tek satırlık gözlem besleme (`step(X_t, Y_obs)`) ve durum telemetrisi alma arayüzü.
  * Plugin Kayıt Mekanizması (`PluginRegistry`): Dış adaptörlerin çalışma zamanında güvenli enjeksiyonu.
  * Derleme ve Dağıtım Betikleri (Shared Library / Cython / Binary Wheel izolasyonu) ve uçtan uca entegrasyon test süiti.

---

## 3. DEĞİŞTİRİLEMEZ KURALLAR (INVARIANTS LİSTESİ)

Bu projede görev alan hiçbir geliştirici veya yapay zeka ajanı, hangi sprintte olunursa olunsun aşağıdaki **7 Temel İnvaryantı** ihlal eden kod yazamaz:

1. **Epistemik Yanılabilirlik İnvaryantı ($P < 1.0$):**
   * Sistemdeki hiçbir kuralın, hipotezin veya uzman ağın ($m_k$) güven katsayısı $1.0$'a eşit veya büyük olamaz ($P \in [P_{min}, P_{max})$ ve $P_{max} < 1.0$).
   * Güven artışı her zaman $P += \eta \cdot (P_{max} - P)$ asimptotik formülüyle hesaplanmak zorundadır.
   * $P \ge 1.0$ değerini atamaya çalışan herhangi bir işlem Domain katmanında `EpistemicInvariantViolationException` fırlatarak anında reddedilir.

2. **Geriye Dönük Regresyon Koruması İnvaryantı (Döngü D Geçiş Şartı):**
   * $H_2$ (Gölge Dal / Karantina) havuzunda sentezlenen hiçbir yeni mikro ağ ($m_{yeni}$) veya değiştirilen hiçbir kural, geçmişte doğrulanmış çapa verileri ($\mathcal{D}_{anchor}$) üzerinde yapılan regresyon ve stabilite testini ($\text{Loss}_{eski} \le \text{Kabul\_Toleransı}$) **geçmeden** $H_3$ (Aktif Çekirdek) havuzuna terfi ettirilemez.
   * Regresyon testi başarısız olan modüller $H_2$'de karantinada tutulur ve sınır şartları daraltılır.

3. **Mutlak Alan Bağımsızlığı İnvaryantı (Domain-Agnosticism):**
   * `Domain` ve `Application` katmanlarında fizik, finans, biyoloji, otomotiv, tıp vb. hiçbir spesifik alana ait sabit terim, sınıf adı, değişken adı veya iş kuralı kullanılamaz.
   * Çekirdek yalnızca soyut matematiksel/epistemik kavramları tanır: Gözlem ($X(t)$), Bileşen Vektörü ($V$), Aktif Alt-Uzay ($V_{aktif}$), Boyut ($\theta$), Güven ($P$), Hata ($\varepsilon$), Kayıp ($\text{Loss}$) ve Havuzlar ($H_1, H_2, H_3$).

4. **Sıfır Harici Bağımlılık İnvaryantı (Pure Domain Rule):**
   * `Domain` katmanı hiçbir üçüncü parti kütüphaneye (ORM, ML framework, HTTP client, JSON parser vb.) bağımlı olamaz. Yalnızca dilin yerleşik standart tiplerini ve saf matematiksel yapılarını kullanır.

5. **Çapraz Konsensüs ve Zehirlenme Koruması İnvaryantı (Döngü C):**
   * Tek bir dış kaynağın (Single Source of Truth yanılgısı) sağladığı bilgiyle global parametre uzayı ($\Omega_{global}$) **asla** genişletilemez.
   * Yeni bir boyutun ($\theta_{kesif}$) $\Omega_{global}$'e kabulü için bağımsız $N$ kaynak ($N \ge N_{min}$) üzerinden yapılan çapraz doğrulamanın konsensüs eşiğini ($C(\text{Kaynaklar}) \ge \text{Eşik}$) aşması zorunludur.

6. **Seyrek Dikkat Kapısı Zorunluluğu ($G_{sparse}$):**
   * Ham veri havuzundan ($H_1$) gelen ayrıştırılmış vektör $V$, doğrudan $H_3$ içindeki modellere beslenemez.
   * Her zaman önce $\Phi(X)$ invaryans maskelemesinden ve $G_{sparse}(V)$ seyrek dikkat kapısından geçerek yalnızca aktif değişim gösteren ($\Delta v_i \neq 0$ veya durum geçişi tetikleyen) $V_{aktif}$ ($\lvert V_{aktif}\rvert \ll \lvert V\rvert$) alt-uzayına indirgenmek zorundadır.

7. **Kapalı Kutu Çekirdek ve Bağımlılıkların Tersine Çevrilmesi (IP & DIP):**
   * `Domain` veya `Application` katmanı, `Infrastructure` katmanındaki hiçbir somut sınıfı doğrudan örnekleyemez (`new` / doğrudan `import` yasaktır). Tüm dış etkileşimler yalnızca `Application` katmanındaki soyut SPI portları üzerinden yürütülür.

---

## 4. MİMARİ KARAR KAYITLARI (ADR ÖZETİ)

* **ADR-001 (Üçlü Havuz İzolasyonu):** Öğrenme sırasında aktif sistemin (Production/Inference) kararsızlığa düşmesini engellemek için Ham Veri ($H_1$), Karantina/Gölge Test ($H_2$) ve Aktif Çekirdek ($H_3$) fiziksel ve mantıksal olarak birbirinden ayrılmıştır.
* **ADR-002 (Kademeli Sorgulama Maliyet Optimizasyonu):** Her anomalide pahalı dış araştırmaya (Katman 4) çıkmak yerine önce iç uzayda ($\Omega_{global}$) pasif boyut taraması (Döngü B - Seviye 2) yapılması zorunlu kılınmıştır. Dış araştırma yalnızca iç uzay yetersiz kaldığında (Seviye 3) tetiklenir.
* **ADR-003 (Monolitik Ağ Yerine Modüler Mikro-Ağ Topolojisi):** Yeni bir boyut ($\theta_{kesif}$) keşfedildiğinde tüm sistemi baştan eğitmek yerine, yalnızca $\text{Dim}(V_{aktif} \cup \{\theta_{kesif}\})$ girdili hafif bir mikro alt-ağ ($m_{yeni}$) sentezlenip Router ağa dinamik kapı (gating) ile bağlanır.