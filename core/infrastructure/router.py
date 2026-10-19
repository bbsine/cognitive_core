"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Modül Yönlendirici (Router / Context Gate).

Bu modül; 3. Havuz (H_3 - Aktif Çekirdek) içerisindeki uzman mikro modüllerin
(M = {m_1, m_2, ..., m_k}) parametre uzayındaki uzmanlık bölgelerini (`centroid_vector`
ve `activation_radius`) yöneten ve gelen girdi vektörünü Euclidean veya Cosine
benzerliğine göre en uygun uzmana yönlendiren `ModuleRouter` sınıfını tanımlar.

İnşa Kuralları:
- Sıfır Harici Kütüphane (Invariant #4): Yalnızca 'math', 'typing', 'dataclasses'.
- Mutlak Alan Bağımsızlığı (Invariant #3): Domain-agnostic tasarım.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Set, Tuple

from core.application.ports import IMemoryRepository
from core.domain import (
    CognitiveDomainException,
    ComponentVector,
    MicroModule,
)


@dataclass
class ModuleRegionProfile:
    """
    Bir H_3 uzman modülünün parametre uzayındaki ilgi/uzmanlık bölgesini
    temsil eden kayıt yapısı.
    """

    module_id: str
    centroid_vector: ComponentVector
    activation_radius: float
    is_fallback: bool = False


class ModuleRouter:
    """
    Katman 2 / H_3 Yönlendirici Topoloji (Router Network / Context Gate).

    Görevleri:
    1. Her H_3 modülünün (`MicroModule`) parametre uzayındaki ilgi/uzmanlık alanını
       belirten `centroid_vector` ve `activation_radius` bilgisini tutar.
    2. Yeni bir girdi vektörü geldiğinde, kütüphanedeki tüm terfi etmiş (H_3)
       modüller arasında Euclidean veya Cosine benzerliği hesaplayarak en yakın/uygun
       modülü `active_module` olarak seçer (`route(vector) -> Optional[MicroModule]`).
    3. Hiçbir modül güven eşiğini veya aktivasyon yarıçapını karşılamıyorsa,
       yapılandırmaya göre en genel (fallback) modülü seçer veya `None` dönerek
       anomali sinyalini tetikler.
    """

    SUPPORTED_METRICS: Tuple[str, ...] = ("euclidean", "cosine")

    def __init__(
        self,
        memory_repo: Optional[IMemoryRepository] = None,
        similarity_metric: str = "euclidean",
        min_confidence_threshold: float = 0.20,
        default_activation_radius: float = 10.0,
        allow_fallback: bool = True,
    ) -> None:
        metric_clean = str(similarity_metric).strip().lower()
        if metric_clean not in self.SUPPORTED_METRICS:
            raise CognitiveDomainException(
                f"ModuleRouter 'similarity_metric' yalnızca {self.SUPPORTED_METRICS} değerlerinden biri olabilir.",
                context={"similarity_metric": similarity_metric},
            )

        if (
            not isinstance(min_confidence_threshold, (int, float))
            or isinstance(min_confidence_threshold, bool)
            or math.isnan(float(min_confidence_threshold))
            or not (0.0 <= float(min_confidence_threshold) < 1.0)
        ):
            raise CognitiveDomainException(
                "ModuleRouter 'min_confidence_threshold' [0.0, 1.0) aralığında olmalıdır.",
                context={"min_confidence_threshold": min_confidence_threshold},
            )

        if (
            not isinstance(default_activation_radius, (int, float))
            or isinstance(default_activation_radius, bool)
            or math.isnan(float(default_activation_radius))
            or float(default_activation_radius) <= 0.0
        ):
            raise CognitiveDomainException(
                "ModuleRouter 'default_activation_radius' pozitif bir sayı (> 0.0) olmalıdır.",
                context={"default_activation_radius": default_activation_radius},
            )

        self.memory_repo: Optional[IMemoryRepository] = memory_repo
        self.similarity_metric: str = metric_clean
        self.min_confidence_threshold: float = float(min_confidence_threshold)
        self.default_activation_radius: float = float(default_activation_radius)
        self.allow_fallback: bool = bool(allow_fallback)

        self._profiles: Dict[str, ModuleRegionProfile] = {}
        self._registered_modules: Dict[str, MicroModule] = {}
        self._active_context_module_id: Optional[str] = None

    @property
    def active_context_module_id(self) -> Optional[str]:
        """Son yönlendirilen veya kilitlenen aktif modül kimliğini döner."""
        return self._active_context_module_id

    def set_active_context_module(self, module_id: Optional[str]) -> None:
        """Aktif rejim modülü bağlam işaretçisini günceller."""
        self._active_context_module_id = module_id

    def register_module(
        self,
        module: MicroModule,
        centroid_vector: Optional[ComponentVector] = None,
        activation_radius: Optional[float] = None,
        is_fallback: bool = False,
    ) -> None:
        """
        Bir H_3 uzman modülünün uzmanlık merkezini (`centroid_vector`) ve
        aktivasyon yarıçapını (`activation_radius`) yönlendiriciye kaydeder.
        """
        if not isinstance(module, MicroModule):
            raise CognitiveDomainException(
                "ModuleRouter.register_module yalnızca bir MicroModule nesnesi kabul eder."
            )

        self._registered_modules[module.module_id] = module

        effective_centroid = (
            centroid_vector if centroid_vector is not None else module.centroid_vector
        )
        effective_radius = (
            float(activation_radius)
            if activation_radius is not None
            else (
                module.activation_radius
                if not math.isinf(module.activation_radius)
                else self.default_activation_radius
            )
        )

        if effective_centroid is not None:
            module.centroid_vector = effective_centroid
            module.activation_radius = effective_radius
            self._profiles[module.module_id] = ModuleRegionProfile(
                module_id=module.module_id,
                centroid_vector=effective_centroid,
                activation_radius=effective_radius,
                is_fallback=is_fallback,
            )

    def unregister_module(self, module_id: str) -> None:
        """Modülü yönlendirici kütüphanesinden ve profillerinden çıkarır."""
        self._profiles.pop(module_id, None)
        self._registered_modules.pop(module_id, None)
        if self._active_context_module_id == module_id:
            self._active_context_module_id = None

    def get_registered_modules(self) -> List[MicroModule]:
        """Yönlendiricide ve bağlı bellek deposunda kayıtlı aktif modülleri döner."""
        if self.memory_repo is not None:
            active_in_repo = {m.module_id: m for m in self.memory_repo.get_active_modules()}
            active_in_repo.update(self._registered_modules)
            return list(active_in_repo.values())
        return list(self._registered_modules.values())

    def get_profile(self, module_id: str) -> Optional[ModuleRegionProfile]:
        """Modülün kayıtlı uzmanlık bölgesi profilini döner."""
        return self._profiles.get(module_id)

    def route(
        self,
        vector: ComponentVector,
        candidates: Optional[List[MicroModule]] = None,
    ) -> Optional[MicroModule]:
        """
        Gelen girdi vektörü için H_3 aktif modül kütüphanesindeki en uygun
        uzman modülü seçer.
        """
        if not isinstance(vector, ComponentVector):
            raise CognitiveDomainException(
                "ModuleRouter.route 'vector' parametresi bir ComponentVector nesnesi olmalıdır.",
                context={"received_type": type(vector).__name__},
            )

        module_pool: List[MicroModule] = (
            list(candidates)
            if candidates is not None
            else (
                self.memory_repo.get_active_modules()
                if self.memory_repo is not None
                else list(self._registered_modules.values())
            )
        )

        compatible_modules: List[MicroModule] = [
            mod
            for mod in module_pool
            if mod.active_dimensions.issubset(vector.dimension_keys)
        ]

        if not compatible_modules:
            return None

        in_radius_specialists: List[Tuple[float, float, int, int, MicroModule]] = []
        general_unbounded_modules: List[Tuple[float, int, MicroModule]] = []
        fallback_candidates: List[Tuple[float, float, int, MicroModule]] = []

        for mod in compatible_modules:
            centroid = self._resolve_centroid(mod)
            radius = self._resolve_radius(mod)
            meets_confidence = mod.confidence.score >= self.min_confidence_threshold
            is_current_context = 1 if mod.module_id == self._active_context_module_id else 0

            if centroid is not None:
                distance, similarity = self._compute_distance_and_similarity(
                    vector=vector,
                    centroid=centroid,
                )
                if meets_confidence and distance <= radius:
                    in_radius_specialists.append(
                        (
                            round(similarity, 6),
                            mod.confidence.score,
                            is_current_context,
                            len(mod.active_dimensions),
                            mod,
                        )
                    )
                else:
                    fallback_candidates.append(
                        (
                            similarity,
                            mod.confidence.score,
                            -len(mod.active_dimensions),
                            mod,
                        )
                    )
            else:
                if meets_confidence:
                    general_unbounded_modules.append(
                        (
                            mod.confidence.score,
                            len(mod.active_dimensions),
                            mod,
                        )
                    )
                else:
                    fallback_candidates.append(
                        (
                            0.5,
                            mod.confidence.score,
                            -len(mod.active_dimensions),
                            mod,
                        )
                    )

        # 1. Öncelik: Aktivasyon yarıçapı içindeki güvenilir rejim uzmanları
        if in_radius_specialists:
            in_radius_specialists.sort(
                key=lambda item: (item[0], item[2], item[1], item[3]),
                reverse=True,
            )
            chosen = in_radius_specialists[0][4]
            self._active_context_module_id = chosen.module_id
            return chosen

        # 2. Öncelik: Henüz bölge kısıtı olmayan genel (unbounded) güvenilir modüller
        if general_unbounded_modules:
            general_unbounded_modules.sort(
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )
            chosen = general_unbounded_modules[0][2]
            self._active_context_module_id = chosen.module_id
            return chosen

        # 3. Öncelik: Güven eşiği veya yarıçap dışı durumda Fallback
        if self.allow_fallback and fallback_candidates:
            fallback_candidates.sort(
                key=lambda item: (item[0], item[1], item[2]),
                reverse=True,
            )
            chosen = fallback_candidates[0][3]
            self._active_context_module_id = chosen.module_id
            return chosen

        return None

    def compute_similarity(
        self,
        vec_a: ComponentVector,
        vec_b: ComponentVector,
        dimensions: Optional[Set[str]] = None,
    ) -> float:
        """İki ComponentVector arasındaki benzerliği [0.0, 1.0] aralığında hesaplar."""
        _, sim = self._compute_distance_and_similarity(vec_a, vec_b, dimensions)
        return sim

    def compute_distance(
        self,
        vec_a: ComponentVector,
        vec_b: ComponentVector,
        dimensions: Optional[Set[str]] = None,
    ) -> float:
        """İki ComponentVector arasındaki mesafeyi (Euclidean veya Cosine) hesaplar."""
        dist, _ = self._compute_distance_and_similarity(vec_a, vec_b, dimensions)
        return dist

    def _resolve_centroid(self, module: MicroModule) -> Optional[ComponentVector]:
        if module.centroid_vector is not None:
            return module.centroid_vector
        profile = self._profiles.get(module.module_id)
        return profile.centroid_vector if profile is not None else None

    def _resolve_radius(self, module: MicroModule) -> float:
        if not math.isinf(module.activation_radius):
            return module.activation_radius
        profile = self._profiles.get(module.module_id)
        if profile is not None:
            return profile.activation_radius
        return float("inf")

    def _compute_distance_and_similarity(
        self,
        vector: ComponentVector,
        centroid: ComponentVector,
        dimensions: Optional[Set[str]] = None,
    ) -> Tuple[float, float]:
        keys: Set[str] = (
            set(dimensions)
            if dimensions is not None
            else (vector.dimension_keys | centroid.dimension_keys)
        )

        if not keys:
            return 0.0, 1.0

        if self.similarity_metric == "cosine":
            dot_prod: float = 0.0
            norm_a_sq: float = 0.0
            norm_b_sq: float = 0.0
            for k in keys:
                va = vector.values.get(k, 0.0)
                vb = centroid.values.get(k, 0.0)
                dot_prod += va * vb
                norm_a_sq += va * va
                norm_b_sq += vb * vb

            if norm_a_sq <= 1e-15 and norm_b_sq <= 1e-15:
                return 0.0, 1.0
            if norm_a_sq <= 1e-15 or norm_b_sq <= 1e-15:
                return 1.0, 0.0

            cos_sim = dot_prod / (math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq))
            cos_sim = max(-1.0, min(1.0, cos_sim))
            normalized_sim = 0.5 * (cos_sim + 1.0)
            distance = 1.0 - cos_sim
            return distance, normalized_sim

        sum_sq: float = 0.0
        for k in keys:
            va = vector.values.get(k, 0.0)
            vb = centroid.values.get(k, 0.0)
            diff = va - vb
            sum_sq += diff * diff

        euclidean_dist = math.sqrt(sum_sq)
        similarity = 1.0 / (1.0 + euclidean_dist)
        return euclidean_dist, similarity


__all__ = [
    "ModuleRegionProfile",
    "ModuleRouter",
]