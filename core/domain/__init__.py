"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Saf Domain Katmanı Paketi.

Bu paket; sıfır harici kütüphane bağımlılığı ve mutlak alan bağımsızlığı
ilkelerine bağlı kalarak çekirdeğin tüm istisna, değer nesnesi, olay ve
varlık sınıflarını dış katmanlara (Application) açar.
"""

from core.domain.entities import (
    GlobalParameterSpace,
    MicroModule,
    Observation,
)
from core.domain.events import (
    AnomalyQuarantinedEvent,
    DimensionDiscoveredEvent,
    DomainEvent,
    ModuleDemotedEvent,
    ModulePromotedEvent,
)
from core.domain.exceptions import (
    CognitiveDomainException,
    DimensionNotFoundException,
    EpistemicInvariantViolationException,
    RegressionFailureException,
)
from core.domain.values import (
    ComponentVector,
    ConfidenceScore,
    DimensionAxis,
    TargetQuery,
)

__all__ = [
    # Exceptions
    "CognitiveDomainException",
    "EpistemicInvariantViolationException",
    "RegressionFailureException",
    "DimensionNotFoundException",
    # Value Objects
    "ConfidenceScore",
    "DimensionAxis",
    "ComponentVector",
    "TargetQuery",
    # Domain Events
    "DomainEvent",
    "AnomalyQuarantinedEvent",
    "DimensionDiscoveredEvent",
    "ModulePromotedEvent",
    "ModuleDemotedEvent",
    # Domain Entities & Aggregates
    "Observation",
    "MicroModule",
    "GlobalParameterSpace",
]