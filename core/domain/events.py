"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Domain Olayları (Domain Events).

Bu modül; epistemik havuzlar (H_1, H_2, H_3) arasındaki durum geçişlerini,
anomali karantina sinyallerini, yeni boyut keşiflerini ve modül terfi/tenzil
olaylarını taşıyan değişmez (immutable) domain olaylarını tanımlar.

İnşa Kuralları:
- Sıfır Harici Kütüphane (Invariant #4)
- Mutlak Alan Bağımsızlığı (Invariant #3)
- Immutability: Tüm olay sınıfları @dataclass(frozen=True) yapısındadır.
"""

from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
import math
from typing import Any, Optional

from core.domain.exceptions import (
    CognitiveDomainException,
    EpistemicInvariantViolationException,
)
from core.domain.values import (
    ConfidenceScore,
    DimensionAxis,
    TargetQuery,
)


@dataclass(frozen=True)
class DomainEvent(ABC):
    """
    Tüm bilişsel domain olayları için soyut temel sınıf.
    """

    event_id: str
    timestamp: float

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise CognitiveDomainException(
                "Domain olayı 'event_id' alanı boş olmayan bir metin (str) olmalıdır.",
                context={"event_id": self.event_id},
            )

        if (
            not isinstance(self.timestamp, (int, float))
            or isinstance(self.timestamp, bool)
            or math.isnan(float(self.timestamp))
            or math.isinf(float(self.timestamp))
            or float(self.timestamp) < 0.0
        ):
            raise CognitiveDomainException(
                "Domain olayı 'timestamp' alanı negatif olmayan geçerli bir sayı olmalıdır.",
                context={"timestamp": self.timestamp},
            )
        object.__setattr__(self, "timestamp", float(self.timestamp))


@dataclass(frozen=True)
class AnomalyQuarantinedEvent(DomainEvent):
    """
    Döngü A'da tahmin hatası (|Y_obs - Y_pred| > ε) tespit edildiğinde,
    ilgili durumun ve modülün H_2 (Karantina) havuzuna taşındığını ve
    Döngü B / C için hedef sorgunun üretildiğini belirten olay.
    """

    module_id: Optional[str]
    target_query: TargetQuery
    observed_outcome: Any

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.module_id is not None and (
            not isinstance(self.module_id, str) or not self.module_id.strip()
        ):
            raise CognitiveDomainException(
                "AnomalyQuarantinedEvent 'module_id' alanı None veya boş olmayan bir metin olmalıdır.",
                context={"module_id": self.module_id},
            )

        if not isinstance(self.target_query, TargetQuery):
            raise CognitiveDomainException(
                "AnomalyQuarantinedEvent 'target_query' alanı bir TargetQuery nesnesi olmalıdır.",
                context={"target_query": self.target_query},
            )

        if self.observed_outcome is None:
            raise CognitiveDomainException(
                "AnomalyQuarantinedEvent 'observed_outcome' (Y_obs) alanı None olamaz."
            )


@dataclass(frozen=True)
class DimensionDiscoveredEvent(DomainEvent):
    """
    Katman 4 (Döngü C: Çapraz Konsensüs) sonucunda yeni bir boyut ekseninin (θ_kesif)
    doğrulanarak global parametre uzayına (Ω_global) eklendiğini belirten olay.
    """

    discovered_dimension: DimensionAxis
    triggering_query: TargetQuery

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.discovered_dimension, DimensionAxis):
            raise CognitiveDomainException(
                "DimensionDiscoveredEvent 'discovered_dimension' alanı bir DimensionAxis nesnesi olmalıdır.",
                context={"discovered_dimension": self.discovered_dimension},
            )

        if not isinstance(self.triggering_query, TargetQuery):
            raise CognitiveDomainException(
                "DimensionDiscoveredEvent 'triggering_query' alanı bir TargetQuery nesnesi olmalıdır.",
                context={"triggering_query": self.triggering_query},
            )


@dataclass(frozen=True)
class ModulePromotedEvent(DomainEvent):
    """
    Katman 5 (Döngü D: Regresyon ve Stabilite Testi) aşamasını başarıyla geçen
    bir mikro alt-ağın gölge daldan (H_2) aktif çekirdeğe (H_3) terfi ettirildiğini
    belirten olay.
    """

    module_id: str
    confidence: ConfidenceScore
    source_pool: str = "H_2"
    target_pool: str = "H_3"

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.module_id, str) or not self.module_id.strip():
            raise CognitiveDomainException(
                "ModulePromotedEvent 'module_id' alanı boş olamaz.",
                context={"module_id": self.module_id},
            )

        if not isinstance(self.confidence, ConfidenceScore):
            raise CognitiveDomainException(
                "ModulePromotedEvent 'confidence' alanı bir ConfidenceScore nesnesi olmalıdır.",
                context={"confidence": self.confidence},
            )

        if not isinstance(self.source_pool, str) or not self.source_pool.strip():
            raise CognitiveDomainException(
                "ModulePromotedEvent 'source_pool' alanı boş olamaz.",
                context={"source_pool": self.source_pool},
            )

        if not isinstance(self.target_pool, str) or not self.target_pool.strip():
            raise CognitiveDomainException(
                "ModulePromotedEvent 'target_pool' alanı boş olamaz.",
                context={"target_pool": self.target_pool},
            )

        if self.source_pool == self.target_pool:
            raise EpistemicInvariantViolationException(
                "Terfi olayında kaynak havuz (source_pool) ve hedef havuz (target_pool) aynı olamaz.",
                context={"source_pool": self.source_pool, "target_pool": self.target_pool},
            )


@dataclass(frozen=True)
class ModuleDemotedEvent(DomainEvent):
    """
    Döngü A'da anomali/çelişki üreten bir uzman modülün güven katsayısının
    düşürülerek (P = P * γ) aktif çekirdekten (H_3) karantina havuzuna (H_2)
    taşındığını belirten olay.
    """

    module_id: str
    confidence: ConfidenceScore
    source_pool: str = "H_3"
    target_pool: str = "H_2"

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.module_id, str) or not self.module_id.strip():
            raise CognitiveDomainException(
                "ModuleDemotedEvent 'module_id' alanı boş olamaz.",
                context={"module_id": self.module_id},
            )

        if not isinstance(self.confidence, ConfidenceScore):
            raise CognitiveDomainException(
                "ModuleDemotedEvent 'confidence' alanı bir ConfidenceScore nesnesi olmalıdır.",
                context={"confidence": self.confidence},
            )

        if not isinstance(self.source_pool, str) or not self.source_pool.strip():
            raise CognitiveDomainException(
                "ModuleDemotedEvent 'source_pool' alanı boş olamaz.",
                context={"source_pool": self.source_pool},
            )

        if not isinstance(self.target_pool, str) or not self.target_pool.strip():
            raise CognitiveDomainException(
                "ModuleDemotedEvent 'target_pool' alanı boş olamaz.",
                context={"target_pool": self.target_pool},
            )

        if self.source_pool == self.target_pool:
            raise EpistemicInvariantViolationException(
                "Tenzil (demote) olayında kaynak havuz (source_pool) ve hedef havuz (target_pool) aynı olamaz.",
                context={"source_pool": self.source_pool, "target_pool": self.target_pool},
            )