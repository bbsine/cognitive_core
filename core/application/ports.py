"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Application Katmanı SPI Port Sözleşmeleri.

Bu modül; kapalı kutu (Closed-Box) bilişsel çekirdeğin dış dünya (veritabanları,
harici araştırma ajanları ve sinir ağı/tensör hesaplama motorları) ile iletişim
kurmasını sağlayan Soyut Servis Sağlayıcı Arayüzlerini (SPI - Service Provider
Interfaces) tanımlar.

İnşa Kuralları:
- Sıfır Harici Kütüphane (Invariant #4): Yalnızca standart kütüphane ('abc', 'typing').
- Mutlak Alan Bağımsızlığı (Invariant #3): Alana özgü hiçbir terim içermez.
- Bağımlılıkların Tersine Çevrilmesi (Invariant #7 - DIP): Application katmanı
  Infrastructure katmanını bilmez; dış adaptörler bu portları implemente eder.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from core.domain import (
    ComponentVector,
    ConfidenceScore,
    DimensionAxis,
    MicroModule,
    Observation,
    TargetQuery,
)


class IMemoryRepository(ABC):
    """
    Epistemik Bellek Havuzları (H_1, H_2, H_3) için soyut veri erişim sözleşmesi.

    Mimari Sorumluluk:
    - H_1 (Ham Veri Giriş Havuzu): Dış gözlemlerin (Observation) tamponlanması
      ve geriye dönük çapa (anchor) / bağlam verisi olarak sorgulanması.
    - H_2 (Karantina ve Gölge Dal Havuzu): Anomali üreten veya henüz regresyon
      testini (Döngü D) geçmemiş aday uzman modüllerin (MicroModule) izolasyonu.
    - H_3 (Aktif Çekirdek Havuzu): Doğrulanmış ve aktif çıkarım yapan uzman
      modüllerin (M = {m_1, m_2, ..., m_k}) kalıcılığı.
    """

    @abstractmethod
    def save_observation(self, observation: Observation) -> None:
        """
        Dış dünyadan gelen yeni bir gözlem varlığını 1. Havuz'a (H_1 - Ham Veri Girişi)
        kaydeder.

        Args:
            observation: Ayrıştırılmış bileşen vektörünü (V) ve gözlemlenen durumu
                (Y_obs) barındıran Observation nesnesi.
        """
        raise NotImplementedError

    @abstractmethod
    def get_recent_observations(self, limit: int) -> List[Observation]:
        """
        1. Havuz'dan (H_1) zamansal olarak en güncel gözlem kayıtlarını getirir.
        Bu veriler iç simülasyon ön-eğitiminde ve Döngü D regresyon/stabilite
        testlerinde çapa verisi (anchor data) olarak kullanılır.

        Args:
            limit: Getirilecek maksimum gözlem sayısı (limit > 0).

        Returns:
            En güncelden eskiye sıralanmış Observation listesi.
        """
        raise NotImplementedError

    @abstractmethod
    def save_quarantined_module(self, module: MicroModule) -> None:
        """
        Anomali nedeniyle tenzil edilen (Döngü A) veya yeni sentezlenen (Katman 5)
        bir uzman modülü 2. Havuz'a (H_2 - Karantina / Gölge Dal) kaydeder veya günceller.

        Args:
            module: Karantinaya alınan veya gölge dalda test edilen MicroModule nesnesi.
        """
        raise NotImplementedError

    @abstractmethod
    def get_quarantined_modules(self) -> List[MicroModule]:
        """
        2. Havuz'da (H_2 - Karantina / Gölge Dal) bekleyen tüm uzman modülleri döner.

        Returns:
            H_2 havuzundaki MicroModule nesnelerinin listesi.
        """
        raise NotImplementedError

    @abstractmethod
    def save_active_module(self, module: MicroModule) -> None:
        """
        Doğrulanmış veya güven katsayısı (ConfidenceScore) güncellenmiş bir uzman
        modülü 3. Havuz'a (H_3 - Aktif Çekirdek) kaydeder veya günceller.

        Args:
            module: H_3 aktif çekirdeğinde görev yapan MicroModule nesnesi (`P < 1.0`).
        """
        raise NotImplementedError

    @abstractmethod
    def get_active_modules(self) -> List[MicroModule]:
        """
        3. Havuz'da (H_3 - Aktif Çekirdek) bulunan ve doğrudan çıkarım (Döngü A)
        için Router ağ tarafından yönlendirilebilecek tüm aktif uzman modülleri döner.

        Returns:
            H_3 havuzundaki aktif MicroModule nesnelerinin listesi.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_quarantined_module(self, module_id: str) -> None:
        """
        Belirtilen kimliğe sahip modülü 2. Havuz'dan (H_2) kaldırır
        (Örn: Modül Döngü D testini geçip H_3'e terfi ettirildiğinde).

        Args:
            module_id: Silinecek modülün benzersiz kimliği.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_active_module(self, module_id: str) -> None:
        """
        Belirtilen kimliğe sahip modülü 3. Havuz'dan (H_3) kaldırır
        (Örn: Modül Döngü A'da anomali üretip H_2 karantinasına taşındığında).

        Args:
            module_id: Aktif çekirdekten çıkarılacak modülün benzersiz kimliği.
        """
        raise NotImplementedError


class IResearchAgent(ABC):
    """
    Katman 4 (Döngü C: Çapraz Dış Araştırma ve Boyut Genişletme) için
    harici bilgi çekimi yapan araştırma ajanlarının SPI sözleşmesi.

    Mimari Sorumluluk:
    İç parametre uzayının (Ω_global) anomalileri açıklamakta yetersiz kaldığı
    durumlarda (Katman 3 / Seviye 3), üretilen karşı-olgusal hedef sorguyu
    Q(Δ, V_aktif, Y_obs) dış veri kaynaklarına (API, Veritabanı, Metin Korpusu vb.)
    yöneltir ve aday boyut hipotezlerini toplar.
    """

    @abstractmethod
    def query_external_sources(self, query: TargetQuery) -> List[Dict[str, Any]]:
        """
        Karşı-olgusal hedef sorguyu (Q) bağımsız dış kaynaklarda araştırır ve
        yapılandırılmış aday boyut/hipotez bulgularını döner.

        Dönen her bir sözlük elemanı, Döngü C konsensüs ve zehirlenme koruması
        denetiminde kullanılmak üzere kaynağın bulgularını (örn. aday boyut ekseni
        tanımı, istatistiksel korelasyon/güven metrikleri ve kaynak kimliği) içerir.

        Args:
            query: Durum değişim izolasyonu (Δ), aktif çalışma vektörü (V_aktif)
                ve gözlemlenen sonucu (Y_obs) içeren TargetQuery nesnesi.

        Returns:
            Bağımsız dış kaynaklardan elde edilen yapılandırılmış hipotez adaylarının listesi.
        """
        raise NotImplementedError


class INeuralEnginePort(ABC):
    """
    Katman 1 (Nitelik Ayrıştırma Φ(X)) ve Katman 5 (Otonom Sinir Ağı Derleme,
    İç Simülasyon Ön-Eğitimi ve Döngü D Regresyon Testi) için hesaplama motoru
    SPI sözleşmesi.

    Mimari Sorumluluk:
    Domain ve Application katmanlarını spesifik makine öğrenmesi/tensör
    kütüphanelerinden (PyTorch, LibTorch, Eigen, ONNX vb.) izole eder.
    """

    @abstractmethod
    def decompose_input(self, raw_input: Dict[str, Any]) -> ComponentVector:
        """
        Katman 1 - Nitelik Ayrıştırma Operatörü Φ(X):
        Dış dünyadan gelen ham gözlem verisini X(t) bileşen tensörlerine
        (V = {v_1, v_2, ..., v_n}) ayrıştırır ve invaryans maskelemesini uygulayarak
        değişmez bir ComponentVector nesnesine dönüştürür.

        Args:
            raw_input: Ham dış gözlem verisini içeren anahtar-değer sözlüğü.

        Returns:
            Ayrıştırılmış ve doğrulanmış ComponentVector (V) nesnesi.
        """
        raise NotImplementedError

    @abstractmethod
    def synthesize_candidate_module(
        self,
        target_query: TargetQuery,
        context_data: List[Observation],
    ) -> MicroModule:
        """
        Katman 5 - Neurogenesis ve İç Simülasyonda Hızlı Ön-Eğitim:
        Yeni keşfedilen/onaylanan boyut ekseni veya anomali bağlamı için
        Dim(V_aktif ∪ {θ_kesif}) girdi uzayında çalışan yeni bir aday mikro alt-ağ
        (m_yeni) şablonu oluşturur ve bağlam verileriyle (context_data) yerel
        optimizasyonunu tamamlar.

        Üretilen modül doğrudan H_3'e eklenmez; önce H_2 gölge dalında
        `evaluate_regression` testine tabi tutulur.

        Args:
            target_query: Anomaliyi tetikleyen karşı-olgusal hedef sorgu (Q).
            context_data: Ön-eğitim ve sınır şartı optimizasyonu için kullanılacak
                geçmiş/bağlamsal Observation listesi.

        Returns:
            H_2 gölge dalında test edilmeye hazır aday MicroModule (m_yeni) nesnesi.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_regression(
        self,
        candidate: MicroModule,
        anchor_data: List[Observation],
        tolerance: float,
    ) -> bool:
        """
        Katman 5 / Döngü D - Regresyon ve Stabilite Testi (Geriye Dönük Koruma):
        H_2 gölge dalında sentezlenen aday modülün (m_yeni), geçmişte doğrulanmış
        çapa verilerini (anchor_data) bozup bozmadığını (Catastrophic Forgetting)
        denetler.

        Matematiksel Kural (Invariant #2):
            Loss_eski <= tolerance (Kabul_Toleransı)
            Bu koşul sağlanmadan hiçbir modül H_3 aktif çekirdeğine terfi ettirilemez.

        Args:
            candidate: Test edilen aday MicroModule (m_yeni) nesnesi.
            anchor_data: Geçmişte doğrulanmış çapa gözlemlerinin (D_anchor) listesi.
            tolerance: İzin verilen maksimum geriye dönük kayıp eşiği (Kabul_Toleransı >= 0.0).

        Returns:
            True: Bozunum yok (Loss_eski <= tolerance), modül H_3'e birleştirilebilir.
            False: Bozunum var (Loss_eski > tolerance), modül H_2'de karantinada tutulmalıdır.
        """
        raise NotImplementedError


__all__ = [
    "IMemoryRepository",
    "IResearchAgent",
    "INeuralEnginePort",
]