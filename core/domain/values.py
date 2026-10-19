"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Domain Değer Nesneleri (Value Objects).

Bu modül; epistemik güven skorunu (P), boyut eksenlerini (θ), ayrıştırılmış
bileşen vektörlerini (V, V_aktif) ve karşı-olgusal hedef sorgularını (Q)
değişmez (immutable) değer nesneleri olarak tanımlar.

İnşa Kuralları:
- Sıfır Harici Kütüphane (Invariant #4)
- Mutlak Alan Bağımsızlığı (Invariant #3)
- Immutability: Tüm değer nesneleri @dataclass(frozen=True) yapısındadır.
- P < 1.0 Kuralı (Invariant #1): 0.0 <= score < 0.9999 sınırları zorunludur.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any, ClassVar, Dict, Set

from core.domain.exceptions import (
    CognitiveDomainException,
    DimensionNotFoundException,
    EpistemicInvariantViolationException,
)


@dataclass(frozen=True)
class ConfidenceScore:
    """
    Epistemik güven katsayısını temsil eden değişmez (immutable) değer nesnesi.

    Matematiksel Sınır:
        P(kural) ∈ [P_min, P_max),  0.0 <= P_min < P_max < 1.0
        Varsayılan yarı-açık aralık: [0.0, 0.9999)
    """

    DEFAULT_P_MIN: ClassVar[float] = 0.0
    DEFAULT_P_MAX: ClassVar[float] = 0.9999

    score: float
    p_min: float = field(default=DEFAULT_P_MIN)
    p_max: float = field(default=DEFAULT_P_MAX)

    def __post_init__(self) -> None:
        if math.isnan(self.score) or math.isinf(self.score):
            raise EpistemicInvariantViolationException(
                "Epistemik güven skoru NaN veya sonsuz (Inf) olamaz.",
                context={"score": self.score},
            )

        if math.isnan(self.p_min) or math.isnan(self.p_max):
            raise EpistemicInvariantViolationException(
                "Epistemik sınır parametreleri (p_min, p_max) NaN olamaz.",
                context={"p_min": self.p_min, "p_max": self.p_max},
            )

        if not (0.0 <= self.p_min < self.p_max < 1.0):
            raise EpistemicInvariantViolationException(
                "Epistemik sınırlar 0.0 <= P_min < P_max < 1.0 koşulunu sağlamalıdır.",
                context={"p_min": self.p_min, "p_max": self.p_max},
            )

        if self.score >= 1.0:
            raise EpistemicInvariantViolationException(
                "Epistemik İnvaryant İhlali: Güven katsayısı P >= 1.0 olamaz (Mutlak kesinlik yasağı).",
                context={"score": self.score, "p_max": self.p_max},
            )

        if not (self.p_min <= self.score < self.p_max):
            raise EpistemicInvariantViolationException(
                f"Güven skoru yarı-açık [{self.p_min}, {self.p_max}) aralığında olmalıdır.",
                context={"score": self.score, "p_min": self.p_min, "p_max": self.p_max},
            )

    def reinforce(self, learning_rate: float) -> ConfidenceScore:
        """
        Döngü A (Başarılı Tahmin / Pekiştirme):
        P_{t+1} = P_t + η * (P_max - P_t),  η ∈ (0.0, 1.0)
        """
        if math.isnan(learning_rate) or not (0.0 < learning_rate < 1.0):
            raise EpistemicInvariantViolationException(
                "Asimptotik öğrenme oranı (η) (0.0, 1.0) açık aralığında olmalıdır.",
                context={"learning_rate": learning_rate},
            )

        raw_next_score = self.score + learning_rate * (self.p_max - self.score)
        max_representable_below_limit = math.nextafter(self.p_max, self.p_min)
        safe_next_score = min(raw_next_score, max_representable_below_limit)

        return ConfidenceScore(
            score=safe_next_score,
            p_min=self.p_min,
            p_max=self.p_max,
        )

    def penalize(self, decay_rate: float) -> ConfidenceScore:
        """
        Döngü A (Anomali / Çelişki):
        P_{t+1} = P_t * γ,  γ ∈ (0.0, 1.0)
        """
        if math.isnan(decay_rate) or not (0.0 < decay_rate < 1.0):
            raise EpistemicInvariantViolationException(
                "Güven düşürme / ceza çarpanı (γ) (0.0, 1.0) açık aralığında olmalıdır.",
                context={"decay_rate": decay_rate},
            )

        raw_next_score = self.score * decay_rate
        safe_next_score = max(self.p_min, raw_next_score)

        return ConfidenceScore(
            score=safe_next_score,
            p_min=self.p_min,
            p_max=self.p_max,
        )


@dataclass(frozen=True)
class DimensionAxis:
    """
    Global parametre uzayındaki (Ω_global) veya girdi vektöründeki
    tek bir soyut boyut eksenini (θ) ve eksik girdi durumunda kullanılacak
    taban değerini (`default_value`) temsil eder.
    """

    name: str
    data_type: str = "continuous"
    variance_threshold: float = 0.0
    default_value: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise CognitiveDomainException(
                "Boyut ekseni adı (name) boş olamaz.",
                context={"name": self.name},
            )

        if not isinstance(self.data_type, str) or not self.data_type.strip():
            raise CognitiveDomainException(
                "Boyut veri tipi (data_type) boş olamaz.",
                context={"data_type": self.data_type},
            )

        if (
            not isinstance(self.variance_threshold, (int, float))
            or isinstance(self.variance_threshold, bool)
            or math.isnan(float(self.variance_threshold))
            or math.isinf(float(self.variance_threshold))
            or float(self.variance_threshold) < 0.0
        ):
            raise CognitiveDomainException(
                "Varyans eşiği (variance_threshold) negatif, NaN veya sonsuz olamaz.",
                context={"variance_threshold": self.variance_threshold},
            )

        if (
            not isinstance(self.default_value, (int, float))
            or isinstance(self.default_value, bool)
            or math.isnan(float(self.default_value))
            or math.isinf(float(self.default_value))
        ):
            raise CognitiveDomainException(
                "Boyut varsayılan taban değeri (default_value) geçerli bir sayı (float) olmalıdır.",
                context={"default_value": self.default_value},
            )

        object.__setattr__(self, "variance_threshold", float(self.variance_threshold))
        object.__setattr__(self, "default_value", float(self.default_value))


@dataclass(frozen=True)
class ComponentVector:
    """
    Nitelik ayrıştırma operatörü Φ(X) tarafından ayrıştırılmış bileşen
    tensörünün (V = {v_1, v_2, ..., v_n}) veya seyrek dikkat kapısından
    geçmiş aktif çalışma alt-uzayının (V_aktif) değişmez temsili.
    """

    values: Dict[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.values, dict):
            raise CognitiveDomainException(
                "ComponentVector 'values' alanı bir sözlük (Dict[str, float]) olmalıdır."
            )

        sanitized: Dict[str, float] = {}
        for key, val in self.values.items():
            if not isinstance(key, str) or not key.strip():
                raise CognitiveDomainException(
                    "Bileşen vektörü boyut anahtarları boş olmayan metin (str) olmalıdır.",
                    context={"invalid_key": key},
                )
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise CognitiveDomainException(
                    f"'{key}' boyutunun değeri sayısal (float) olmalıdır.",
                    context={"key": key, "value": val},
                )
            float_val = float(val)
            if math.isnan(float_val) or math.isinf(float_val):
                raise CognitiveDomainException(
                    f"'{key}' boyutunun değeri NaN veya sonsuz (Inf) olamaz.",
                    context={"key": key, "value": float_val},
                )
            sanitized[key] = float_val

        object.__setattr__(self, "values", sanitized)

    @property
    def dimension_keys(self) -> Set[str]:
        """Vektördeki mevcut tüm boyut isimlerinin kümesini döndürür."""
        return set(self.values.keys())

    @property
    def dimensionality(self) -> int:
        """Vektörün boyut sayısını (|V|) döndürür."""
        return len(self.values)

    def extract_subspace(self, active_keys: Set[str]) -> ComponentVector:
        """
        Seyrek Dikkat Kapısı (G_sparse) tarafından seçilen aktif boyutlara göre
        alt-uzay vektörünü (V_aktif ⊆ V) filtreler ve yeni bir ComponentVector döndürür.
        """
        missing_keys = set(active_keys) - self.values.keys()
        if missing_keys:
            raise DimensionNotFoundException(
                "Alt-uzay çıkarımı sırasında mevcut vektörde bulunmayan boyut(lar) talep edildi.",
                context={
                    "missing_dimensions": sorted(missing_keys),
                    "available_dimensions": sorted(self.values.keys()),
                },
            )

        subspace_dict: Dict[str, float] = {
            key: self.values[key] for key in active_keys
        }
        return ComponentVector(values=subspace_dict)

    def calculate_delta(self, other: ComponentVector) -> Dict[str, float]:
        """
        Mevcut vektör ile bir diğer bileşen vektörü arasındaki anlık değişimi
        (Δv_i = self[v_i] - other[v_i]) hesaplar.
        """
        if not isinstance(other, ComponentVector):
            raise CognitiveDomainException(
                "Delta hesaplaması yalnızca bir diğer ComponentVector ile yapılabilir."
            )

        missing_in_other = self.values.keys() - other.values.keys()
        if missing_in_other:
            raise DimensionNotFoundException(
                "Delta hesaplaması için referans alınan vektörde eksik boyut(lar) mevcut.",
                context={"missing_dimensions": sorted(missing_in_other)},
            )

        return {
            key: self.values[key] - other.values[key]
            for key in self.values
        }


@dataclass(frozen=True)
class TargetQuery:
    """
    Katman 3 (Seviye 3: Karşı-Olgusal Teşhis) tarafından iç uzay yetersiz
    kaldığında üretilen ve Katman 4'teki dış araştırma ajanlarına iletilen
    hedef sorgu nesnesi: Q(Δ, V_aktif, Y_obs).
    """

    delta_state: Dict[str, float]
    active_vector: ComponentVector
    observed_outcome: Any

    def __post_init__(self) -> None:
        if not isinstance(self.delta_state, dict) or len(self.delta_state) == 0:
            raise CognitiveDomainException(
                "TargetQuery 'delta_state' (Δ) alanı boş olmayan bir sözlük olmalıdır."
            )

        sanitized_delta: Dict[str, float] = {}
        for key, val in self.delta_state.items():
            if not isinstance(key, str) or not key.strip():
                raise CognitiveDomainException(
                    "Delta durum anahtarları geçerli bir boyut adı (str) olmalıdır.",
                    context={"invalid_key": key},
                )
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise CognitiveDomainException(
                    f"Delta durumundaki '{key}' değeri sayısal (float) olmalıdır.",
                    context={"key": key, "value": val},
                )
            float_val = float(val)
            if math.isnan(float_val) or math.isinf(float_val):
                raise CognitiveDomainException(
                    f"Delta durumundaki '{key}' değeri NaN veya sonsuz olamaz.",
                    context={"key": key, "value": float_val},
                )
            sanitized_delta[key] = float_val

        if not isinstance(self.active_vector, ComponentVector):
            raise CognitiveDomainException(
                "TargetQuery 'active_vector' (V_aktif) bir ComponentVector nesnesi olmalıdır."
            )

        if self.observed_outcome is None:
            raise CognitiveDomainException(
                "TargetQuery 'observed_outcome' (Y_obs) None olamaz."
            )

        object.__setattr__(self, "delta_state", sanitized_delta)