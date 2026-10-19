"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Application Katmanı Orkestrasyon Servisleri.

Bu modül; 5 katmanlı bilişsel mimarinin ve 3 epistemik havuzun (H_1, H_2, H_3)
iş akışlarını (Use Cases) koordine eden CognitiveOrchestrator sınıfını tanımlar.

İnşa Kuralları:
- Sıfır Harici Kütüphane (Invariant #4): Yalnızca Python standart kütüphanesi.
- Mutlak Alan Bağımsızlığı (Invariant #3): Domain-agnostic tasarım.
- Bağımlılıkların Tersine Çevrilmesi (Invariant #7): SPI portları üzerinden etkileşim.
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional, Set
import uuid

from core.application.ports import (
    IMemoryRepository,
    INeuralEnginePort,
    IResearchAgent,
)
from core.domain import (
    AnomalyQuarantinedEvent,
    CognitiveDomainException,
    ComponentVector,
    EpistemicInvariantViolationException,
    GlobalParameterSpace,
    MicroModule,
    ModulePromotedEvent,
    Observation,
    RegressionFailureException,
    TargetQuery,
)


class CognitiveOrchestrator:
    """
    Bilişsel Çekirdek Motoru'nun ana kullanım senaryolarını (Use Cases) ve
    A, B, D bilişsel döngülerini orkestre eden uygulama servisi.
    """

    def __init__(
        self,
        memory_repo: IMemoryRepository,
        neural_engine: INeuralEnginePort,
        parameter_space: GlobalParameterSpace,
        research_agent: Optional[IResearchAgent] = None,
        router: Optional[Any] = None,
    ) -> None:
        if not isinstance(memory_repo, IMemoryRepository):
            raise CognitiveDomainException(
                "CognitiveOrchestrator 'memory_repo' parametresi IMemoryRepository arayüzünü implemente etmelidir.",
                context={"received_type": type(memory_repo).__name__},
            )
        if not isinstance(neural_engine, INeuralEnginePort):
            raise CognitiveDomainException(
                "CognitiveOrchestrator 'neural_engine' parametresi INeuralEnginePort arayüzünü implemente etmelidir.",
                context={"received_type": type(neural_engine).__name__},
            )
        if not isinstance(parameter_space, GlobalParameterSpace):
            raise CognitiveDomainException(
                "CognitiveOrchestrator 'parameter_space' parametresi bir GlobalParameterSpace nesnesi olmalıdır.",
                context={"received_type": type(parameter_space).__name__},
            )
        if research_agent is not None and not isinstance(research_agent, IResearchAgent):
            raise CognitiveDomainException(
                "CognitiveOrchestrator 'research_agent' parametresi None veya IResearchAgent olmalıdır.",
                context={"received_type": type(research_agent).__name__},
            )

        self.memory_repo: IMemoryRepository = memory_repo
        self.neural_engine: INeuralEnginePort = neural_engine
        self.parameter_space: GlobalParameterSpace = parameter_space
        self.research_agent: Optional[IResearchAgent] = research_agent
        self.router: Optional[Any] = router

    def infer_and_adapt(
        self,
        raw_input: Dict[str, Any],
        observed_outcome: Optional[Any] = None,
        error_threshold: float = 0.05,
        learning_rate: float = 0.1,
        decay_rate: float = 0.5,
        *,
        defer_quarantine: bool = False,
        record_observation: bool = True,
    ) -> Dict[str, Any]:
        """
        Katman 1, 2 ve Döngü A (Doğrudan Model Tahmini, Pekiştirme veya Karantina)
        iş akışını uçtan uca yürütür.
        """
        if not isinstance(raw_input, dict) or len(raw_input) == 0:
            raise CognitiveDomainException(
                "infer_and_adapt 'raw_input' parametresi boş olmayan bir sözlük olmalıdır."
            )

        if (
            not isinstance(error_threshold, (int, float))
            or isinstance(error_threshold, bool)
            or math.isnan(float(error_threshold))
            or math.isinf(float(error_threshold))
            or float(error_threshold) < 0.0
        ):
            raise CognitiveDomainException(
                "Hata eşiği (error_threshold / ε) negatif olmayan geçerli bir sayı olmalıdır.",
                context={"error_threshold": error_threshold},
            )

        # 1. Nitelik Ayrıştırma Φ(X) ve Eksik Boyut İmpütasyonu (align_vector)
        raw_decomposed: ComponentVector = self.neural_engine.decompose_input(raw_input)
        decomposed_vector: ComponentVector = self.parameter_space.align_vector(
            raw_decomposed.values
        )

        prior_observations: List[Observation] = self.memory_repo.get_recent_observations(limit=1)

        current_observation = Observation(
            id=str(uuid.uuid4()),
            features=decomposed_vector,
            observed_outcome=observed_outcome,
            timestamp=time.time(),
        )
        if record_observation:
            self.memory_repo.save_observation(current_observation)

        # 2. H_3 Aktif Çekirdekten Router (Context Gate) ile en uygun uzman modülü seç
        selected_module: Optional[MicroModule] = self._select_compatible_active_module(
            decomposed_vector
        )

        if selected_module is None:
            if observed_outcome is not None:
                delta_state = self._compute_state_delta(decomposed_vector, prior_observations)
                target_query = TargetQuery(
                    delta_state=delta_state,
                    active_vector=decomposed_vector,
                    observed_outcome=observed_outcome,
                )
                anomaly_event = AnomalyQuarantinedEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    module_id=None,
                    target_query=target_query,
                    observed_outcome=observed_outcome,
                )
                return {
                    "status": "NO_ACTIVE_MODULE_ANOMALY",
                    "prediction": None,
                    "observed_outcome": observed_outcome,
                    "error": None,
                    "module_id": None,
                    "confidence": None,
                    "active_vector": decomposed_vector,
                    "observation": current_observation,
                    "target_query": target_query,
                    "events": [anomaly_event],
                }

            return {
                "status": "NO_ACTIVE_MODULE",
                "prediction": None,
                "observed_outcome": None,
                "error": None,
                "module_id": None,
                "confidence": None,
                "active_vector": decomposed_vector,
                "observation": current_observation,
                "target_query": None,
                "events": [],
            }

        active_vector: ComponentVector = decomposed_vector.extract_subspace(
            selected_module.active_dimensions
        )
        prediction: Any = selected_module.evaluate(active_vector)

        if observed_outcome is None:
            return {
                "status": "INFERRED",
                "prediction": prediction,
                "observed_outcome": None,
                "error": None,
                "module_id": selected_module.module_id,
                "confidence": selected_module.confidence,
                "active_vector": active_vector,
                "observation": current_observation,
                "target_query": None,
                "events": [],
            }

        prediction_error: float = self._calculate_prediction_error(observed_outcome, prediction)

        if prediction_error <= float(error_threshold):
            updated_confidence = selected_module.reinforce_confidence(learning_rate)
            selected_module.update_centroid(decomposed_vector)
            self.memory_repo.save_active_module(selected_module)

            return {
                "status": "REINFORCED",
                "prediction": prediction,
                "observed_outcome": observed_outcome,
                "error": prediction_error,
                "module_id": selected_module.module_id,
                "confidence": updated_confidence,
                "active_vector": active_vector,
                "observation": current_observation,
                "target_query": None,
                "events": [],
            }

        # Tahmin hatası eşiği aştı: |Y_obs - Y_pred| > ε
        delta_state = self._compute_state_delta(active_vector, prior_observations)
        target_query = TargetQuery(
            delta_state=delta_state,
            active_vector=active_vector,
            observed_outcome=observed_outcome,
        )

        # Histerezis Kalkanı aktifse modülü hemen silme
        if defer_quarantine:
            return {
                "status": "ANOMALY_DETECTED",
                "prediction": prediction,
                "observed_outcome": observed_outcome,
                "error": prediction_error,
                "module_id": selected_module.module_id,
                "selected_module": selected_module,
                "confidence": selected_module.confidence,
                "active_vector": active_vector,
                "observation": current_observation,
                "target_query": target_query,
                "events": [],
            }

        demoted_event = selected_module.demote_to_quarantine(
            decay_rate=decay_rate,
            source_pool="H_3",
            target_pool="H_2",
        )
        self.memory_repo.remove_active_module(selected_module.module_id)
        self.memory_repo.save_quarantined_module(selected_module)

        anomaly_event = AnomalyQuarantinedEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            module_id=selected_module.module_id,
            target_query=target_query,
            observed_outcome=observed_outcome,
        )

        return {
            "status": "ANOMALY_QUARANTINED",
            "prediction": prediction,
            "observed_outcome": observed_outcome,
            "error": prediction_error,
            "module_id": selected_module.module_id,
            "confidence": selected_module.confidence,
            "active_vector": active_vector,
            "observation": current_observation,
            "target_query": target_query,
            "events": [demoted_event, anomaly_event],
        }

    def diagnose_internal_space(
        self,
        target_query: TargetQuery,
        observation_window: int = 50,
    ) -> Dict[str, Any]:
        """
        Katman 3 / Döngü B (Seviye 2: İç Bellek & Graf Taraması):
        Global parametre uzayındaki pasif boyutları tarar.
        """
        if not isinstance(target_query, TargetQuery):
            raise CognitiveDomainException(
                "diagnose_internal_space 'target_query' parametresi bir TargetQuery nesnesi olmalıdır.",
                context={"received_type": type(target_query).__name__},
            )

        if observation_window <= 0:
            raise CognitiveDomainException(
                "diagnose_internal_space 'observation_window' pozitif bir tam sayı olmalıdır.",
                context={"observation_window": observation_window},
            )

        passive_dimensions: Set[str] = self.parameter_space.get_passive_dimensions(
            target_query.active_vector
        )

        if not passive_dimensions:
            return {
                "status": "ESCALATE_TO_EXTERNAL",
                "explained_internally": False,
                "passive_dimensions": set(),
                "candidate_dimensions": set(),
                "target_query": target_query,
            }

        recent_observations: List[Observation] = self.memory_repo.get_recent_observations(
            limit=observation_window
        )

        explanatory_candidates: Set[str] = set()
        for dim_name in passive_dimensions:
            axis = self.parameter_space.get_dimension(dim_name)
            observed_values: List[float] = [
                obs.features.values[dim_name]
                for obs in recent_observations
                if dim_name in obs.features.values
            ]

            if not observed_values:
                continue

            if len(observed_values) >= 2:
                spread = max(observed_values) - min(observed_values)
                if spread > axis.variance_threshold:
                    explanatory_candidates.add(dim_name)
            else:
                if abs(observed_values[0] - axis.default_value) > axis.variance_threshold:
                    explanatory_candidates.add(dim_name)

        if not explanatory_candidates:
            return {
                "status": "ESCALATE_TO_EXTERNAL",
                "explained_internally": False,
                "passive_dimensions": passive_dimensions,
                "candidate_dimensions": set(),
                "target_query": target_query,
            }

        return {
            "status": "EXPLAINED_INTERNALLY",
            "explained_internally": True,
            "passive_dimensions": passive_dimensions,
            "candidate_dimensions": explanatory_candidates,
            "target_query": target_query,
        }

    def promote_shadow_module(
        self,
        candidate: MicroModule,
        anchor_sample_size: int = 50,
        tolerance: float = 0.05,
        anchor_override: Optional[List[Observation]] = None,
    ) -> ModulePromotedEvent:
        """
        Katman 5 / Döngü D (Regresyon ve Stabilite Testi & Topolojik Birleştirme).
        """
        if not isinstance(candidate, MicroModule):
            raise CognitiveDomainException(
                "promote_shadow_module 'candidate' parametresi bir MicroModule nesnesi olmalıdır.",
                context={"received_type": type(candidate).__name__},
            )

        if not isinstance(anchor_sample_size, int) or isinstance(anchor_sample_size, bool) or anchor_sample_size <= 0:
            raise CognitiveDomainException(
                "promote_shadow_module 'anchor_sample_size' pozitif bir tam sayı olmalıdır.",
                context={"anchor_sample_size": anchor_sample_size},
            )

        if (
            not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or math.isnan(float(tolerance))
            or math.isinf(float(tolerance))
            or float(tolerance) < 0.0
        ):
            raise CognitiveDomainException(
                "promote_shadow_module 'tolerance' negatif olmayan geçerli bir sayı olmalıdır.",
                context={"tolerance": tolerance},
            )

        quarantined_ids = {m.module_id for m in self.memory_repo.get_quarantined_modules()}
        if candidate.module_id not in quarantined_ids:
            self.memory_repo.save_quarantined_module(candidate)

        anchor_data: List[Observation] = (
            list(anchor_override)
            if anchor_override is not None
            else self.memory_repo.get_recent_observations(limit=anchor_sample_size)
        )

        passed_regression: bool = self.neural_engine.evaluate_regression(
            candidate=candidate,
            anchor_data=anchor_data,
            tolerance=float(tolerance),
        )

        if not passed_regression:
            raise RegressionFailureException(
                f"Döngü D Regresyon Testi Başarısız: '{candidate.module_id}' aday modülü "
                f"çapa verilerinde bozunuma yol açtı (Tolerans={tolerance}). Modül H_2 karantinasında tutuldu.",
                context={
                    "module_id": candidate.module_id,
                    "anchor_count": len(anchor_data),
                    "tolerance": tolerance,
                },
            )

        self.memory_repo.remove_quarantined_module(candidate.module_id)
        self.memory_repo.save_active_module(candidate)

        if self.router is not None and hasattr(self.router, "register_module"):
            self.router.register_module(candidate)

        return candidate.promote_to_active(source_pool="H_2", target_pool="H_3")

    def _select_compatible_active_module(
        self,
        vector: ComponentVector,
    ) -> Optional[MicroModule]:
        active_modules: List[MicroModule] = self.memory_repo.get_active_modules()
        if not active_modules:
            return None

        if self.router is not None and hasattr(self.router, "route"):
            routed: Optional[MicroModule] = self.router.route(vector, active_modules)
            if routed is not None:
                return routed

        compatible_modules: List[MicroModule] = [
            mod
            for mod in active_modules
            if mod.active_dimensions.issubset(vector.dimension_keys)
        ]

        if not compatible_modules:
            return None

        compatible_modules.sort(
            key=lambda m: (m.confidence.score, len(m.active_dimensions)),
            reverse=True,
        )
        return compatible_modules[0]

    @staticmethod
    def _calculate_prediction_error(observed: Any, predicted: Any) -> float:
        if (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and isinstance(predicted, (int, float))
            and not isinstance(predicted, bool)
        ):
            error = abs(float(observed) - float(predicted))
            if math.isnan(error) or math.isinf(error):
                raise EpistemicInvariantViolationException(
                    "Tahmin hatası hesaplaması NaN veya sonsuz (Inf) üretti.",
                    context={"observed": observed, "predicted": predicted},
                )
            return error

        if isinstance(observed, dict) and isinstance(predicted, dict):
            common_keys = set(observed.keys()) & set(predicted.keys())
            if not common_keys:
                return 1.0
            diffs = [
                abs(float(observed[k]) - float(predicted[k]))
                for k in common_keys
                if isinstance(observed[k], (int, float))
                and isinstance(predicted[k], (int, float))
            ]
            return sum(diffs) / len(diffs) if diffs else (0.0 if observed == predicted else 1.0)

        return 0.0 if observed == predicted else 1.0

    @staticmethod
    def _compute_state_delta(
        active_vector: ComponentVector,
        prior_observations: List[Observation],
    ) -> Dict[str, float]:
        if not prior_observations:
            return dict(active_vector.values)

        last_features = prior_observations[0].features
        if active_vector.dimension_keys.issubset(last_features.dimension_keys):
            last_subspace = last_features.extract_subspace(active_vector.dimension_keys)
            return active_vector.calculate_delta(last_subspace)

        delta: Dict[str, float] = {}
        for key, current_val in active_vector.values.items():
            prev_val = last_features.values.get(key, 0.0)
            delta[key] = current_val - prev_val
        return delta