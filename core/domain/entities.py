"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Domain Varlıkları ve Kökleri (Entities & Aggregates).

Bu modül; H_1 ham veri gözlemlerini (Observation), H_2/H_3 uzman mikro ağlarını
(MicroModule) ve global parametre uzayını (GlobalParameterSpace - Ω_global)
yaşam döngüsü ve kimliği olan domain varlıkları olarak tanımlar.

İnşa Kuralları:
- Sıfır Harici Kütüphane (Invariant #4)
- Mutlak Alan Bağımsızlığı (Invariant #3)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

from core.domain.events import (
    ModuleDemotedEvent,
    ModulePromotedEvent,
)
from core.domain.exceptions import (
    CognitiveDomainException,
    DimensionNotFoundException,
    EpistemicInvariantViolationException,
)
from core.domain.values import (
    ComponentVector,
    ConfidenceScore,
    DimensionAxis,
)


@dataclass
class Observation:
    """
    1. Havuz (H_1 - Ham Veri Girişi) için dış gözlem kayıt varlığı.
    """

    id: str
    features: ComponentVector
    observed_outcome: Optional[Any] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise CognitiveDomainException(
                "Observation 'id' alanı boş olmayan bir metin (str) olmalıdır.",
                context={"id": self.id},
            )

        if not isinstance(self.features, ComponentVector):
            raise CognitiveDomainException(
                "Observation 'features' alanı bir ComponentVector nesnesi olmalıdır.",
                context={"features": self.features},
            )

        if (
            not isinstance(self.timestamp, (int, float))
            or isinstance(self.timestamp, bool)
            or math.isnan(float(self.timestamp))
            or math.isinf(float(self.timestamp))
            or float(self.timestamp) < 0.0
        ):
            raise CognitiveDomainException(
                "Observation 'timestamp' alanı negatif olmayan geçerli bir sayı olmalıdır.",
                context={"timestamp": self.timestamp},
            )
        self.timestamp = float(self.timestamp)


@dataclass
class MicroModule:
    """
    H_3 (Aktif Çekirdek) veya H_2 (Gölge Dal / Karantina) havuzundaki
    uzman mikro alt-ağ modülünü (m_k) temsil eden domain varlığı.

    Çoklu Rejim Yönlendirme (Context Gate) Alanları:
    - `centroid_vector`: Modülün parametre uzayındaki uzmanlık merkezini (rejim ağırlık merkezi) tutar.
    - `activation_radius`: Modülün merkez etrafındaki geçerli aktivasyon yarıçapını belirtir.
    - `reinforcement_count`: Modülün kendi rejiminde kaç kez doğrulandığını izler.
    """

    module_id: str
    active_dimensions: Set[str]
    confidence: ConfidenceScore
    predict_fn: Callable[[ComponentVector], Any]
    exceptions: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    centroid_vector: Optional[ComponentVector] = None
    activation_radius: float = float("inf")
    reinforcement_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.module_id, str) or not self.module_id.strip():
            raise CognitiveDomainException(
                "MicroModule 'module_id' alanı boş olmayan bir metin (str) olmalıdır.",
                context={"module_id": self.module_id},
            )

        if not isinstance(self.active_dimensions, set) or len(self.active_dimensions) == 0:
            raise CognitiveDomainException(
                "MicroModule 'active_dimensions' alanı boş olmayan bir boyut kümesi (Set[str]) olmalıdır.",
                context={"active_dimensions": self.active_dimensions},
            )

        for dim_name in self.active_dimensions:
            if not isinstance(dim_name, str) or not dim_name.strip():
                raise CognitiveDomainException(
                    "MicroModule 'active_dimensions' içindeki her boyut adı geçerli bir metin olmalıdır.",
                    context={"invalid_dimension": dim_name},
                )

        self.active_dimensions = set(self.active_dimensions)

        if not isinstance(self.confidence, ConfidenceScore):
            raise CognitiveDomainException(
                "MicroModule 'confidence' alanı bir ConfidenceScore nesnesi olmalıdır.",
                context={"confidence": self.confidence},
            )

        if not callable(self.predict_fn):
            raise CognitiveDomainException(
                "MicroModule 'predict_fn' alanı çağrılabilir (Callable) bir fonksiyon olmalıdır."
            )

        if not isinstance(self.exceptions, list):
            raise CognitiveDomainException(
                "MicroModule 'exceptions' alanı bir liste (List[Dict[str, Any]]) olmalıdır."
            )

        validated_exceptions: List[Dict[str, Any]] = []
        for exc_rule in self.exceptions:
            if not isinstance(exc_rule, dict) or "condition" not in exc_rule or "outcome" not in exc_rule:
                raise CognitiveDomainException(
                    "Her istisna kuralı 'condition' ve 'outcome' anahtarlarını içeren bir sözlük olmalıdır.",
                    context={"invalid_rule": exc_rule},
                )
            if not callable(exc_rule["condition"]):
                raise CognitiveDomainException(
                    "İstisna kuralındaki 'condition' alanı çağrılabilir (Callable) olmalıdır."
                )
            validated_exceptions.append(dict(exc_rule))
        self.exceptions = validated_exceptions

        if (
            not isinstance(self.created_at, (int, float))
            or isinstance(self.created_at, bool)
            or math.isnan(float(self.created_at))
            or math.isinf(float(self.created_at))
            or float(self.created_at) < 0.0
        ):
            raise CognitiveDomainException(
                "MicroModule 'created_at' alanı negatif olmayan geçerli bir zaman damgası olmalıdır.",
                context={"created_at": self.created_at},
            )
        self.created_at = float(self.created_at)

        if self.centroid_vector is not None and not isinstance(self.centroid_vector, ComponentVector):
            raise CognitiveDomainException(
                "MicroModule 'centroid_vector' alanı None veya bir ComponentVector nesnesi olmalıdır.",
                context={"centroid_vector": self.centroid_vector},
            )

        if (
            not isinstance(self.activation_radius, (int, float))
            or isinstance(self.activation_radius, bool)
            or math.isnan(float(self.activation_radius))
            or float(self.activation_radius) <= 0.0
        ):
            raise CognitiveDomainException(
                "MicroModule 'activation_radius' alanı pozitif bir sayı (> 0.0) olmalıdır.",
                context={"activation_radius": self.activation_radius},
            )
        self.activation_radius = float(self.activation_radius)

        if not isinstance(self.reinforcement_count, int) or isinstance(self.reinforcement_count, bool) or self.reinforcement_count < 0:
            raise CognitiveDomainException(
                "MicroModule 'reinforcement_count' negatif olmayan bir tam sayı olmalıdır.",
                context={"reinforcement_count": self.reinforcement_count},
            )

    def evaluate(self, vector: ComponentVector) -> Any:
        """
        Verilen çalışma vektörü (V_aktif) üzerinde modül tahminini (Y_pred) üretir.
        """
        if not isinstance(vector, ComponentVector):
            raise CognitiveDomainException(
                "MicroModule.evaluate yalnızca bir ComponentVector nesnesi kabul eder.",
                context={"received_type": type(vector).__name__},
            )

        missing_dims = self.active_dimensions - vector.dimension_keys
        if missing_dims:
            raise DimensionNotFoundException(
                f"'{self.module_id}' modülünün değerlendirmesi için gerekli aktif boyutlar vektörde eksik.",
                context={
                    "module_id": self.module_id,
                    "missing_dimensions": sorted(missing_dims),
                    "available_dimensions": sorted(vector.dimension_keys),
                },
            )

        for rule in self.exceptions:
            condition_fn: Callable[[ComponentVector], bool] = rule["condition"]
            if condition_fn(vector):
                return rule["outcome"]

        return self.predict_fn(vector)

    def update_centroid(
        self,
        vector: ComponentVector,
        alpha: float = 0.20,
        default_radius: float = 10.0,
    ) -> ComponentVector:
        """
        Modülün uzmanlık merkezi vektörünü (`centroid_vector`) üstel hareketli ortalama
        (EMA) ile günceller. İlk kez çağrıldığında doğrudan girdi vektörünü merkez alır.
        """
        if not isinstance(vector, ComponentVector):
            raise CognitiveDomainException(
                "update_centroid 'vector' parametresi bir ComponentVector olmalıdır."
            )

        clamped_alpha = max(0.01, min(1.0, float(alpha)))
        if self.centroid_vector is None:
            self.centroid_vector = ComponentVector(values=dict(vector.values))
            if math.isinf(self.activation_radius):
                self.activation_radius = max(0.5, float(default_radius))
            return self.centroid_vector

        all_keys = self.centroid_vector.dimension_keys | vector.dimension_keys
        updated_values: Dict[str, float] = {}
        for key in all_keys:
            old_val = self.centroid_vector.values.get(key, 0.0)
            new_val = vector.values.get(key, old_val)
            updated_values[key] = (1.0 - clamped_alpha) * old_val + clamped_alpha * new_val

        self.centroid_vector = ComponentVector(values=updated_values)
        return self.centroid_vector

    def attach_exception(
        self,
        condition: Callable[[ComponentVector], bool],
        outcome: Any,
    ) -> None:
        """Modüle yerel bir alt-kural / istisna kuralı ekler."""
        if not callable(condition):
            raise CognitiveDomainException(
                "attach_exception 'condition' parametresi çağrılabilir (Callable[[ComponentVector], bool]) olmalıdır."
            )

        if outcome is None:
            raise CognitiveDomainException(
                "attach_exception 'outcome' parametresi None olamaz."
            )

        self.exceptions.append(
            {
                "condition": condition,
                "outcome": outcome,
                "attached_at": time.time(),
            }
        )

    def reinforce_confidence(self, learning_rate: float) -> ConfidenceScore:
        """
        Döngü A pekiştirme adımında modülün güven katsayısını asimptotik olarak artırır
        ve başarılı pekiştirme sayacını günceller.
        """
        self.confidence = self.confidence.reinforce(learning_rate)
        self.reinforcement_count += 1
        return self.confidence

    def demote_to_quarantine(
        self,
        decay_rate: float,
        source_pool: str = "H_3",
        target_pool: str = "H_2",
    ) -> ModuleDemotedEvent:
        """
        Döngü A'da anomali tespit edildiğinde modülün güvenini γ çarpanı ile
        düşürür (P = P * γ) ve H_2 karantina havuzuna tenzil olayını döner.
        """
        self.confidence = self.confidence.penalize(decay_rate)
        return ModuleDemotedEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            module_id=self.module_id,
            confidence=self.confidence,
            source_pool=source_pool,
            target_pool=target_pool,
        )

    def promote_to_active(
        self,
        source_pool: str = "H_2",
        target_pool: str = "H_3",
    ) -> ModulePromotedEvent:
        """
        Döngü D regresyon ve stabilite testini geçen modülün H_3 aktif çekirdek
        havuzuna terfi olayını üretir.
        """
        return ModulePromotedEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            module_id=self.module_id,
            confidence=self.confidence,
            source_pool=source_pool,
            target_pool=target_pool,
        )


@dataclass
class GlobalParameterSpace:
    """
    Global Parametre Uzayını (Ω_global) yöneten Aggregate Root sınıfı.

    Sistemde keşfedilmiş ve onaylanmış tüm soyut boyut eksenlerini (DimensionAxis)
    barındırır, eksik girdi boyutlarını eksenlerin taban değerine (`default_value`)
    hizalar (`align_vector`) ve Katman 3 (Döngü B) için pasif boyut taraması sağlar.
    """

    dimensions: Dict[str, DimensionAxis] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.dimensions, dict):
            raise CognitiveDomainException(
                "GlobalParameterSpace 'dimensions' alanı bir sözlük (Dict[str, DimensionAxis]) olmalıdır."
            )

        sanitized: Dict[str, DimensionAxis] = {}
        for key, axis in self.dimensions.items():
            if not isinstance(axis, DimensionAxis):
                raise CognitiveDomainException(
                    "GlobalParameterSpace içindeki her eleman bir DimensionAxis nesnesi olmalıdır.",
                    context={"key": key, "value": axis},
                )
            if key != axis.name:
                raise CognitiveDomainException(
                    "Boyut sözlük anahtarı ile DimensionAxis.name alanı birebir eşleşmelidir.",
                    context={"dict_key": key, "axis_name": axis.name},
                )
            sanitized[key] = axis

        self.dimensions = sanitized

    def register_dimension(self, dim: DimensionAxis) -> None:
        """
        Global parametre uzayına yeni bir boyut ekseni ekler:
        Ω_global = Ω_global ∪ {θ_kesif}
        """
        if not isinstance(dim, DimensionAxis):
            raise CognitiveDomainException(
                "register_dimension yalnızca bir DimensionAxis nesnesi kabul eder.",
                context={"received": dim},
            )

        if dim.name in self.dimensions:
            existing = self.dimensions[dim.name]
            raise EpistemicInvariantViolationException(
                f"Global parametre uzayında '{dim.name}' boyut ekseni zaten kayıtlı (Boyut çakışması).",
                context={
                    "dimension_name": dim.name,
                    "existing_dimension": existing,
                    "new_dimension": dim,
                },
            )

        self.dimensions[dim.name] = dim

    def get_dimension(self, name: str) -> DimensionAxis:
        """İsmi verilen boyut eksenini döner; mevcut değilse DimensionNotFoundException fırlatır."""
        if name not in self.dimensions:
            raise DimensionNotFoundException(
                f"Global parametre uzayında (Ω_global) '{name}' adlı boyut ekseni bulunamadı.",
                context={
                    "requested_dimension": name,
                    "registered_dimensions": sorted(self.dimensions.keys()),
                },
            )
        return self.dimensions[name]

    def has_dimension(self, name: str) -> bool:
        """Boyut ekseninin Ω_global içinde kayıtlı olup olmadığını kontrol eder."""
        return name in self.dimensions

    def align_vector(self, input_dict: Dict[str, Any]) -> ComponentVector:
        """
        Gelen girdi sözlüğünü Global Parametre Uzayına (Ω_global) hizalar (Imputation):
        1. Ω_global içinde kayıtlı olan her boyut ekseni için:
           - Girdide geçerli bir sayısal değer varsa o değeri kullanır.
           - Girdide ilgili boyut eksikse veya None ise hata fırlatmak yerine
             o eksenin `default_value` (varsayılan 0.0) taban değerini atar.
        2. Girdide yer alan diğer geçerli sayısal (int/float, bool olmayan) boyutları
           da kaybetmeden vektöre dahil eder.
        """
        if not isinstance(input_dict, dict):
            raise CognitiveDomainException(
                "GlobalParameterSpace.align_vector 'input_dict' parametresi bir sözlük (dict) olmalıdır.",
                context={"received_type": type(input_dict).__name__},
            )

        aligned_values: Dict[str, float] = {}

        # 1. Kayıtlı tüm Ω_global eksenlerini tarayarak eksik olanları default_value ile doldur
        for dim_name, axis in self.dimensions.items():
            raw_val = input_dict.get(dim_name, None)
            if (
                raw_val is not None
                and isinstance(raw_val, (int, float))
                and not isinstance(raw_val, bool)
            ):
                float_val = float(raw_val)
                if math.isnan(float_val) or math.isinf(float_val):
                    raise CognitiveDomainException(
                        f"'{dim_name}' boyutunun girdi değeri NaN veya sonsuz (Inf) olamaz.",
                        context={"dimension": dim_name, "value": raw_val},
                    )
                aligned_values[dim_name] = float_val
            else:
                aligned_values[dim_name] = axis.default_value

        # 2. Girdide bulunan ancak henüz Ω_global'e kaydedilmemiş geçerli sayısal boyutları koru
        for key, raw_val in input_dict.items():
            if not isinstance(key, str) or not key.strip() or key in aligned_values:
                continue
            if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
                continue
            float_val = float(raw_val)
            if math.isnan(float_val) or math.isinf(float_val):
                raise CognitiveDomainException(
                    f"'{key}' boyutunun girdi değeri NaN veya sonsuz (Inf) olamaz.",
                    context={"dimension": key, "value": raw_val},
                )
            aligned_values[key] = float_val

        if not aligned_values:
            raise CognitiveDomainException(
                "align_vector sonucunda hiçbir geçerli veya kayıtlı boyut elde edilemedi.",
                context={"raw_keys": list(input_dict.keys())},
            )

        return ComponentVector(values=aligned_values)

    def get_passive_dimensions(self, active_vector: ComponentVector) -> Set[str]:
        """
        Katman 3 / Döngü B (Seviye 2: İç Bellek & Graf Taraması):
        Global parametre uzayında (Ω_global) mevcut olup o anki aktif çalışma
        vektöründe (V_aktif) yer almayan pasif boyutların (Ω_global \\ V_aktif)
        isim kümesini döner.
        """
        if not isinstance(active_vector, ComponentVector):
            raise CognitiveDomainException(
                "get_passive_dimensions 'active_vector' parametresi bir ComponentVector olmalıdır.",
                context={"received_type": type(active_vector).__name__},
            )

        return set(self.dimensions.keys()) - active_vector.dimension_keys