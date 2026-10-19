"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Bellek Havuzu Altyapı Adaptörü.

Bu modül; Application katmanındaki IMemoryRepository SPI portunu gerçekleyen,
iş parçacığı güvenli (thread-safe), bellek içi (In-Memory) ve isteğe bağlı
JSON disk kalıcılığı (persistence) destekli somut bellek adaptörünü tanımlar.

Mimari Havuz Yapısı:
- H_1 (Ham Veri Giriş Havuzu): Eklenme sırasını koruyan Observation listesi.
- H_2 (Karantina ve Gölge Dal Havuzu): module_id anahtarlı MicroModule sözlüğü.
- H_3 (Aktif Çekirdek Havuzu): module_id anahtarlı doğrulanmış MicroModule sözlüğü.

İnşa Kuralları:
- Sıfır Harici Kütüphane (Invariant #4): Yalnızca 'typing', 'json', 'pathlib', 'threading'.
- Mutlak Alan Bağımsızlığı (Invariant #3): Domain-agnostic tasarım.
"""

from __future__ import annotations
import json
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional, Union

from core.application.ports import IMemoryRepository
from core.domain import (
    CognitiveDomainException,
    ComponentVector,
    ConfidenceScore,
    MicroModule,
    Observation,
)


class InMemoryMemoryRepository(IMemoryRepository):
    """
    H_1, H_2 ve H_3 epistemik bellek havuzlarını RAM üzerinde iş parçacığı güvenli
    (thread-safe) olarak yöneten ve JSON formatında diske serileştirme/yükleme
    (snapshot dump/load) yeteneği sunan IMemoryRepository adaptörü.
    """

    def __init__(
        self,
        storage_path: Optional[Union[str, Path]] = None,
        auto_persist: bool = False,
    ) -> None:
        """
        Args:
            storage_path: Durum serileştirmesi (JSON dump/load) için varsayılan
                dosya yolu (opsiyonel).
            auto_persist: True ise her yazma/silme işleminden sonra durumu
                otomatik olarak `storage_path` dosyasına kaydeder.
        """
        self._lock: threading.RLock = threading.RLock()
        self._h1_observations: List[Observation] = []
        self._h2_quarantine: Dict[str, MicroModule] = {}
        self._h3_active: Dict[str, MicroModule] = {}

        # JSON'dan geri yükleme (deserialization) sırasında modüllerin çağrılabilir
        # predict_fn fonksiyonlarını yeniden eşleştirmek için çalışma zamanı kayıt defteri.
        self._predictor_registry: Dict[str, Callable[[ComponentVector], Any]] = {}

        self._storage_path: Optional[Path] = (
            Path(storage_path) if storage_path is not None else None
        )
        self._auto_persist: bool = bool(auto_persist)

    # =========================================================================
    # 1. HAVUZ (H_1): HAM VERİ GİRİŞİ (OBSERVATIONS) OPERASYONLARI
    # =========================================================================

    def save_observation(self, observation: Observation) -> None:
        """
        Dış dünyadan gelen gözlem varlığını eklenme sırasını koruyarak H_1 havuzuna kaydeder.
        """
        if not isinstance(observation, Observation):
            raise CognitiveDomainException(
                "InMemoryMemoryRepository.save_observation yalnızca bir Observation nesnesi kabul eder.",
                context={"received_type": type(observation).__name__},
            )

        with self._lock:
            self._h1_observations.append(observation)
            self._trigger_auto_persist()

    def get_recent_observations(self, limit: int) -> List[Observation]:
        """
        H_1 havuzundaki gözlemleri en güncelden (son eklenenden) eskiye doğru
        belirtilen limit dahilinde döndürür.
        """
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise CognitiveDomainException(
                "InMemoryMemoryRepository.get_recent_observations 'limit' parametresi tam sayı (int) olmalıdır.",
                context={"limit": limit},
            )

        if limit <= 0:
            return []

        with self._lock:
            # Son eklenen gözlem listenin sonunda yer aldığı için ters çevirerek dilimle
            return list(reversed(self._h1_observations[-limit:]))

    # =========================================================================
    # 2. HAVUZ (H_2): KARANTİNA VE GÖLGE DAL (QUARANTINE / SHADOW) OPERASYONLARI
    # =========================================================================

    def save_quarantined_module(self, module: MicroModule) -> None:
        """
        Anomali üreten veya gölge dalda test edilen modülü H_2 havuzuna kaydeder/günceller.
        """
        if not isinstance(module, MicroModule):
            raise CognitiveDomainException(
                "InMemoryMemoryRepository.save_quarantined_module yalnızca bir MicroModule nesnesi kabul eder.",
                context={"received_type": type(module).__name__},
            )

        with self._lock:
            self._h2_quarantine[module.module_id] = module
            self._predictor_registry[module.module_id] = module.predict_fn
            self._trigger_auto_persist()

    def get_quarantined_modules(self) -> List[MicroModule]:
        """
        H_2 karantina/gölge dal havuzundaki tüm modüllerin kopyasını liste olarak döner.
        """
        with self._lock:
            return list(self._h2_quarantine.values())

    def remove_quarantined_module(self, module_id: str) -> None:
        """
        Belirtilen kimliğe sahip modülü H_2 havuzundan kaldırır.
        Modül mevcut değilse hata fırlatmadan güvenli şekilde geçer.
        """
        if not isinstance(module_id, str) or not module_id.strip():
            return

        with self._lock:
            removed = self._h2_quarantine.pop(module_id, None)
            if removed is not None:
                self._trigger_auto_persist()

    # =========================================================================
    # 3. HAVUZ (H_3): AKTİF ÇEKİRDEK (ACTIVE CORE) OPERASYONLARI
    # =========================================================================

    def save_active_module(self, module: MicroModule) -> None:
        """
        Doğrulanmış uzman modülü H_3 aktif çekirdek havuzuna kaydeder/günceller.
        """
        if not isinstance(module, MicroModule):
            raise CognitiveDomainException(
                "InMemoryMemoryRepository.save_active_module yalnızca bir MicroModule nesnesi kabul eder.",
                context={"received_type": type(module).__name__},
            )

        with self._lock:
            self._h3_active[module.module_id] = module
            self._predictor_registry[module.module_id] = module.predict_fn
            self._trigger_auto_persist()

    def get_active_modules(self) -> List[MicroModule]:
        """
        H_3 aktif çekirdek havuzundaki tüm uzman modüllerin listesini döner.
        """
        with self._lock:
            return list(self._h3_active.values())

    def remove_active_module(self, module_id: str) -> None:
        """
        Belirtilen kimliğe sahip modülü H_3 aktif çekirdek havuzundan kaldırır.
        Modül mevcut değilse hata fırlatmadan güvenli şekilde geçer.
        """
        if not isinstance(module_id, str) or not module_id.strip():
            return

        with self._lock:
            removed = self._h3_active.pop(module_id, None)
            if removed is not None:
                self._trigger_auto_persist()

    # =========================================================================
    # HAVUZ YÖNETİMİ VE JSON KALICILIK (SNAPSHOT PERSISTENCE) OPERASYONLARI
    # =========================================================================

    def clear_all(self) -> None:
        """
        Tüm epistemik havuzları (H_1, H_2, H_3) ve çalışma zamanı önbelleğini
        atomik olarak temizler.
        """
        with self._lock:
            self._h1_observations.clear()
            self._h2_quarantine.clear()
            self._h3_active.clear()
            self._predictor_registry.clear()
            self._trigger_auto_persist()

    def register_predictor(
        self,
        module_id: str,
        predict_fn: Callable[[ComponentVector], Any],
    ) -> None:
        """
        JSON'dan geri yüklenen modüllerin tahmin fonksiyonlarını (predict_fn)
        modül kimliğiyle eşleştirmek için kayıt defterine ekler.
        """
        if not isinstance(module_id, str) or not module_id.strip():
            raise CognitiveDomainException("Geçerli bir 'module_id' belirtilmelidir.")
        if not callable(predict_fn):
            raise CognitiveDomainException("'predict_fn' çağrılabilir (Callable) olmalıdır.")

        with self._lock:
            self._predictor_registry[module_id] = predict_fn

    def dump_to_file(self, file_path: Optional[Union[str, Path]] = None) -> Path:
        """
        H_1, H_2 ve H_3 havuzlarının anlık durumunu JSON formatında diske yazar.

        Args:
            file_path: Hedef JSON dosya yolu. Verilmezse kurucudaki `storage_path` kullanılır.

        Returns:
            Yazılan dosyanın Path nesnesi.
        """
        with self._lock:
            target_path = Path(file_path) if file_path is not None else self._storage_path
            if target_path is None:
                raise CognitiveDomainException(
                    "JSON durum kaydı için geçerli bir 'file_path' veya 'storage_path' belirtilmelidir."
                )

            target_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot = self._serialize_state()
            serialized_json = json.dumps(snapshot, ensure_ascii=False, indent=2)
            target_path.write_text(serialized_json, encoding="utf-8")
            return target_path

    def load_from_file(
        self,
        file_path: Optional[Union[str, Path]] = None,
        predictor_registry: Optional[Dict[str, Callable[[ComponentVector], Any]]] = None,
    ) -> None:
        """
        Diskteki JSON durum dosyasından H_1, H_2 ve H_3 havuzlarını belleğe geri yükler.

        Args:
            file_path: Okunacak JSON dosya yolu. Verilmezse kurucudaki `storage_path` kullanılır.
            predictor_registry: Modül kimliklerine göre atanacak özel `predict_fn`
                fonksiyon eşlemeleri (opsiyonel). Eşleşme bulunamazsa kayıt defterindeki
                veya varsayılan doğrusal toplama fonksiyonu atanır.
        """
        with self._lock:
            target_path = Path(file_path) if file_path is not None else self._storage_path
            if target_path is None or not target_path.exists():
                raise CognitiveDomainException(
                    "Yüklenecek JSON durum dosyası bulunamadı.",
                    context={"file_path": str(target_path) if target_path else None},
                )

            if predictor_registry:
                for mod_id, fn in predictor_registry.items():
                    self.register_predictor(mod_id, fn)

            raw_text = target_path.read_text(encoding="utf-8")
            try:
                snapshot: Dict[str, Any] = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise CognitiveDomainException(
                    "Bellek havuzu JSON dosyası bozuk veya geçersiz formatta.",
                    context={"file_path": str(target_path), "error": str(exc)},
                ) from exc

            self._deserialize_state(snapshot)

    # =========================================================================
    # DAHİLİ SERİLEŞTİRME YARDIMCILARI (INTERNAL SERIALIZATION HELPERS)
    # =========================================================================

    def _trigger_auto_persist(self) -> None:
        """Auto-persist aktifse ve storage_path tanımlıysa durumu diske yazar."""
        if self._auto_persist and self._storage_path is not None:
            self.dump_to_file(self._storage_path)

    def _serialize_state(self) -> Dict[str, Any]:
        """H_1, H_2 ve H_3 havuzlarını JSON uyumlu sözlük yapısına dönüştürür."""
        return {
            "H_1_observations": [
                {
                    "id": obs.id,
                    "features": dict(obs.features.values),
                    "observed_outcome": obs.observed_outcome,
                    "timestamp": obs.timestamp,
                }
                for obs in self._h1_observations
            ],
            "H_2_quarantine": {
                mod_id: self._serialize_module(mod)
                for mod_id, mod in self._h2_quarantine.items()
            },
            "H_3_active": {
                mod_id: self._serialize_module(mod)
                for mod_id, mod in self._h3_active.items()
            },
        }

    @staticmethod
    def _serialize_module(module: MicroModule) -> Dict[str, Any]:
        """Tek bir MicroModule nesnesinin üst verilerini JSON sözlüğüne dönüştürür."""
        return {
            "module_id": module.module_id,
            "active_dimensions": sorted(module.active_dimensions),
            "confidence": {
                "score": module.confidence.score,
                "p_min": module.confidence.p_min,
                "p_max": module.confidence.p_max,
            },
            "created_at": module.created_at,
        }

    def _deserialize_state(self, snapshot: Dict[str, Any]) -> None:
        """JSON sözlüğünden H_1, H_2 ve H_3 havuz nesnelerini yeniden inşa eder."""
        if not isinstance(snapshot, dict):
            raise CognitiveDomainException("Geçersiz snapshot veri yapısı.")

        reconstructed_h1: List[Observation] = []
        for raw_obs in snapshot.get("H_1_observations", []):
            reconstructed_h1.append(
                Observation(
                    id=raw_obs["id"],
                    features=ComponentVector(values=raw_obs["features"]),
                    observed_outcome=raw_obs.get("observed_outcome"),
                    timestamp=float(raw_obs["timestamp"]),
                )
            )

        reconstructed_h2: Dict[str, MicroModule] = {}
        for mod_id, raw_mod in snapshot.get("H_2_quarantine", {}).items():
            reconstructed_h2[mod_id] = self._deserialize_module(raw_mod)

        reconstructed_h3: Dict[str, MicroModule] = {}
        for mod_id, raw_mod in snapshot.get("H_3_active", {}).items():
            reconstructed_h3[mod_id] = self._deserialize_module(raw_mod)

        self._h1_observations = reconstructed_h1
        self._h2_quarantine = reconstructed_h2
        self._h3_active = reconstructed_h3

    def _deserialize_module(self, raw_mod: Dict[str, Any]) -> MicroModule:
        """JSON verisinden MicroModule nesnesini yeniden oluşturur."""
        mod_id: str = raw_mod["module_id"]
        conf_data: Dict[str, float] = raw_mod["confidence"]

        confidence = ConfidenceScore(
            score=float(conf_data["score"]),
            p_min=float(conf_data.get("p_min", ConfidenceScore.DEFAULT_P_MIN)),
            p_max=float(conf_data.get("p_max", ConfidenceScore.DEFAULT_P_MAX)),
        )

        predict_fn = self._predictor_registry.get(
            mod_id,
            lambda vec: sum(vec.values.values()),
        )

        return MicroModule(
            module_id=mod_id,
            active_dimensions=set(raw_mod["active_dimensions"]),
            confidence=confidence,
            predict_fn=predict_fn,
            created_at=float(raw_mod["created_at"]),
        )


__all__ = ["InMemoryMemoryRepository"]