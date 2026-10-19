"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - SDK Facade ve EngineBuilder.

Bu modül; dış sistemlerin iç katman (Domain, Application, Infrastructure)
karmaşıklığına doğrudan temas etmeden tüm bilişsel yaşam döngüsünü yürütmesini
sağlayan `CognitiveCoreEngine` (Facade) ve 5 Builder Kanununa tabi `EngineBuilder`
sınıflarını tanımlar.

Builder Kanunları (The 5 Invariants of EngineBuilder):
- Kanun #1: Sıfır Konfigürasyon ile Kararlı Başlangıç (Batteries-Included Default)
- Kanun #2: Mutlak Alan Bağımsızlığı (Domain-Agnostic Parameter Space)
- Kanun #3: Polimorfik Doğrulama Ajanı (Swappable Oracle Injection)
- Kanun #4: Akıcı Arayüz ve Doğrulama Kalkanı (Fluent API & Pre-flight Validation)
- Kanun #5: Çift Yönlü Modül ve Uzay Senkronizasyonu
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional, Set, Union
import uuid

from core.application import (
    CognitiveOrchestrator,
    IMemoryRepository,
    INeuralEnginePort,
    IResearchAgent,
)
from core.domain import (
    AnomalyQuarantinedEvent,
    CognitiveDomainException,
    ComponentVector,
    DimensionAxis,
    DimensionDiscoveredEvent,
    DomainEvent,
    EpistemicInvariantViolationException,
    GlobalParameterSpace,
    MicroModule,
    Observation,
    RegressionFailureException,
    TargetQuery,
)
from core.infrastructure import (
    InMemoryMemoryRepository,
    ModuleRouter,
    PurePythonNeuralEngine,
)
from core.infrastructure.research_agent import (
    CrossConsensusEngine,
    MockMultiSourceResearchAgent,
    SourceProvider,
)


class CognitiveCoreEngine:
    """
    5 Katmanlı ve 3 Epistemik Havuzlu (H_1, H_2, H_3) Otonom Bilişsel Çekirdek
    Motoru'nun birleşik istemci arayüzü (SDK Facade).
    """

    def __init__(
        self,
        memory_repo: IMemoryRepository,
        neural_engine: INeuralEnginePort,
        research_agent: IResearchAgent,
        consensus_engine: CrossConsensusEngine,
        parameter_space: GlobalParameterSpace,
        router: Optional[ModuleRouter] = None,
        *,
        error_threshold: float = 0.05,
        learning_rate: float = 0.10,
        decay_rate: float = 0.50,
        consensus_threshold: float = 0.70,
        regression_tolerance: float = 0.05,
        anchor_sample_size: int = 50,
        sparse_delta_threshold: float = 0.0,
        anomaly_patience: int = 3,
    ) -> None:
        self.memory_repo: IMemoryRepository = memory_repo
        self.neural_engine: INeuralEnginePort = neural_engine
        self.research_agent: IResearchAgent = research_agent
        self.consensus_engine: CrossConsensusEngine = consensus_engine
        self.parameter_space: GlobalParameterSpace = parameter_space
        self.router: ModuleRouter = (
            router if router is not None else ModuleRouter(memory_repo=memory_repo)
        )
        if self.router.memory_repo is None:
            self.router.memory_repo = memory_repo

        self.error_threshold: float = float(error_threshold)
        self.learning_rate: float = float(learning_rate)
        self.decay_rate: float = float(decay_rate)
        self.consensus_threshold: float = float(consensus_threshold)
        self.regression_tolerance: float = float(regression_tolerance)
        self.anchor_sample_size: int = int(anchor_sample_size)
        self.sparse_delta_threshold: float = float(sparse_delta_threshold)
        self.anomaly_patience: int = int(anomaly_patience)

        self._consecutive_anomalies: int = 0
        self._anomaly_window: List[Observation] = []

        self.orchestrator: CognitiveOrchestrator = CognitiveOrchestrator(
            memory_repo=self.memory_repo,
            neural_engine=self.neural_engine,
            parameter_space=self.parameter_space,
            research_agent=self.research_agent,
            router=self.router,
        )

    @property
    def dimension_defaults(self) -> Dict[str, float]:
        """GlobalParameterSpace içindeki eksenlerin taban değerlerini döner."""
        return {
            name: getattr(axis, "default_value", 0.0)
            for name, axis in self.parameter_space.dimensions.items()
        }

    def step(
        self,
        raw_input: Dict[str, Any],
        observed_outcome: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not isinstance(raw_input, dict) or len(raw_input) == 0:
            raise CognitiveDomainException("raw_input boş olmayan bir dict olmalıdır.")

        prior_observations: List[Observation] = self.memory_repo.get_recent_observations(limit=1)
        raw_decomposed: ComponentVector = self.neural_engine.decompose_input(raw_input)
        decomposed_vector: ComponentVector = self.parameter_space.align_vector(raw_decomposed.values)
        sparse_active_vector: ComponentVector = self.apply_sparse_attention_gate(
            decomposed_vector, prior_observations
        )

        active_h3_modules = self.memory_repo.get_active_modules()
        has_established_regime = any(
            m.reinforcement_count >= self.anomaly_patience for m in active_h3_modules
        )
        use_hysteresis = (self.anomaly_patience > 1) and has_established_regime

        loop_a_result: Dict[str, Any] = self.orchestrator.infer_and_adapt(
            raw_input=decomposed_vector.values,
            observed_outcome=observed_outcome,
            error_threshold=self.error_threshold,
            learning_rate=self.learning_rate,
            decay_rate=self.decay_rate,
            defer_quarantine=use_hysteresis,
            record_observation=not use_hysteresis,
        )

        loops_executed: List[str] = ["LOOP_A"]
        emitted_events: List[DomainEvent] = list(loop_a_result.get("events", []))
        status: str = loop_a_result["status"]
        evaluated_module_id: Optional[str] = loop_a_result.get("module_id")
        active_module_id: Optional[str] = (
            evaluated_module_id if status in ("INFERRED", "REINFORCED") else None
        )
        promoted_module_id: Optional[str] = None
        discovered_dimension: Optional[str] = None
        consensus_score: Optional[float] = None
        current_confidence = loop_a_result.get("confidence")

        if status in ("INFERRED", "REINFORCED", "NO_ACTIVE_MODULE"):
            if status == "REINFORCED":
                self._consecutive_anomalies = 0
                self._anomaly_window.clear()
                if use_hysteresis:
                    self.memory_repo.save_observation(loop_a_result["observation"])

            return {
                "status": status,
                "prediction": loop_a_result.get("prediction"),
                "observed_outcome": observed_outcome,
                "error": loop_a_result.get("error"),
                "loop_executed": "LOOP_A",
                "loops_executed": loops_executed,
                "active_module_id": active_module_id,
                "evaluated_module_id": evaluated_module_id,
                "promoted_module_id": None,
                "discovered_dimension": None,
                "consensus_score": None,
                "decomposed_vector": decomposed_vector,
                "sparse_active_vector": sparse_active_vector,
                "confidence": current_confidence,
                "consecutive_anomalies": self._consecutive_anomalies,
                "events": emitted_events,
            }

        current_observation: Observation = loop_a_result["observation"]
        target_query: TargetQuery = loop_a_result["target_query"]

        if use_hysteresis and status == "ANOMALY_DETECTED":
            matching_standby = self._find_matching_h3_specialist(
                decomposed_vector, observed_outcome, exclude_module_id=evaluated_module_id
            )
            if matching_standby is not None:
                self._consecutive_anomalies = 0
                self._anomaly_window.clear()
                self.router.set_active_context_module(matching_standby.module_id)
                standby_subspace = decomposed_vector.extract_subspace(matching_standby.active_dimensions)
                standby_pred = matching_standby.evaluate(standby_subspace)
                standby_err = abs(float(observed_outcome) - float(standby_pred))
                updated_conf = matching_standby.reinforce_confidence(self.learning_rate)
                self.memory_repo.save_active_module(matching_standby)
                self.memory_repo.save_observation(current_observation)

                return {
                    "status": "REINFORCED",
                    "prediction": standby_pred,
                    "observed_outcome": observed_outcome,
                    "error": standby_err,
                    "loop_executed": "LOOP_A",
                    "loops_executed": ["LOOP_A"],
                    "active_module_id": matching_standby.module_id,
                    "evaluated_module_id": matching_standby.module_id,
                    "promoted_module_id": None,
                    "discovered_dimension": None,
                    "consensus_score": None,
                    "decomposed_vector": decomposed_vector,
                    "sparse_active_vector": sparse_active_vector,
                    "confidence": updated_conf,
                    "consecutive_anomalies": 0,
                    "events": [],
                }

            self._consecutive_anomalies += 1
            self._anomaly_window.append(current_observation)

            if self._consecutive_anomalies < self.anomaly_patience:
                return {
                    "status": "HYSTERESIS_SHIELDED",
                    "prediction": loop_a_result.get("prediction"),
                    "observed_outcome": observed_outcome,
                    "error": loop_a_result.get("error"),
                    "loop_executed": "LOOP_A",
                    "loops_executed": ["LOOP_A"],
                    "active_module_id": evaluated_module_id,
                    "evaluated_module_id": evaluated_module_id,
                    "promoted_module_id": None,
                    "discovered_dimension": None,
                    "consensus_score": None,
                    "decomposed_vector": decomposed_vector,
                    "sparse_active_vector": sparse_active_vector,
                    "confidence": current_confidence,
                    "consecutive_anomalies": self._consecutive_anomalies,
                    "events": [],
                }

            for buffered_obs in self._anomaly_window:
                self.memory_repo.save_observation(buffered_obs)

            selected_mod: Optional[MicroModule] = loop_a_result.get("selected_module")
            if selected_mod is not None:
                if selected_mod.reinforcement_count < self.anomaly_patience:
                    demoted_ev = selected_mod.demote_to_quarantine(self.decay_rate)
                    self.memory_repo.remove_active_module(selected_mod.module_id)
                    self.memory_repo.save_quarantined_module(selected_mod)
                    emitted_events.append(demoted_ev)

            emitted_events.append(
                AnomalyQuarantinedEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    module_id=evaluated_module_id,
                    target_query=target_query,
                    observed_outcome=observed_outcome,
                )
            )

        regime_context_window: List[Observation] = (
            list(self._anomaly_window) if self._anomaly_window else [current_observation]
        )
        self._consecutive_anomalies = 0
        self._anomaly_window.clear()

        # LOOP_B
        loop_b_result: Dict[str, Any] = self.orchestrator.diagnose_internal_space(
            target_query=target_query,
            observation_window=max(len(regime_context_window), self.anchor_sample_size),
        )
        loops_executed.append("LOOP_B")

        if loop_b_result["status"] == "EXPLAINED_INTERNALLY":
            candidate_dims: Set[str] = loop_b_result["candidate_dimensions"]
            expanded_dims = (
                target_query.active_vector.dimension_keys
                | sparse_active_vector.dimension_keys
                | candidate_dims
            )
            promoted_mod, loop_d_events, loop_d_passed = self._compile_and_promote_module(
                target_query=target_query,
                current_observation=current_observation,
                target_dimensions=expanded_dims,
                regime_observations=regime_context_window if len(regime_context_window) >= 2 else None,
            )
            loops_executed.append("LOOP_D")
            emitted_events.extend(loop_d_events)

            if loop_d_passed and promoted_mod is not None:
                active_module_id = promoted_mod.module_id
                promoted_module_id = promoted_mod.module_id
                current_confidence = promoted_mod.confidence
                self.router.set_active_context_module(promoted_mod.module_id)
                status = "EXPLAINED_INTERNALLY"
            else:
                status = "INTERNAL_REGRESSION_FAILED_QUARANTINED"

            return {
                "status": status,
                "prediction": loop_a_result.get("prediction"),
                "observed_outcome": observed_outcome,
                "error": loop_a_result.get("error"),
                "loop_executed": loops_executed[-1],
                "loops_executed": loops_executed,
                "active_module_id": active_module_id,
                "evaluated_module_id": evaluated_module_id,
                "promoted_module_id": promoted_module_id,
                "discovered_dimension": None,
                "consensus_score": None,
                "decomposed_vector": decomposed_vector,
                "sparse_active_vector": sparse_active_vector,
                "confidence": current_confidence,
                "consecutive_anomalies": 0,
                "events": emitted_events,
            }

        # LOOP_C
        loops_executed.append("LOOP_C")
        external_responses: List[Dict[str, Any]] = self.research_agent.query_external_sources(
            target_query
        )
        is_verified, confirmed_dim_name, consensus_score = (
            self.consensus_engine.evaluate_consensus(
                candidate_responses=external_responses,
                consensus_threshold=self.consensus_threshold,
            )
        )

        if (not is_verified or confirmed_dim_name is None) and len(regime_context_window) >= self.anomaly_patience and self.anomaly_patience > 1:
            target_dimensions_fallback = (
                set(target_query.active_vector.dimension_keys)
                | set(sparse_active_vector.dimension_keys)
            )
            promoted_mod, loop_d_events, loop_d_passed = self._compile_and_promote_module(
                target_query=target_query,
                current_observation=current_observation,
                target_dimensions=target_dimensions_fallback,
                regime_observations=regime_context_window,
            )
            loops_executed.append("LOOP_D")
            emitted_events.extend(loop_d_events)

            if loop_d_passed and promoted_mod is not None:
                self.router.set_active_context_module(promoted_mod.module_id)
                return {
                    "status": "PROMOTED_TO_H3",
                    "prediction": loop_a_result.get("prediction"),
                    "observed_outcome": observed_outcome,
                    "error": loop_a_result.get("error"),
                    "loop_executed": "LOOP_D",
                    "loops_executed": loops_executed,
                    "active_module_id": promoted_mod.module_id,
                    "evaluated_module_id": evaluated_module_id,
                    "promoted_module_id": promoted_mod.module_id,
                    "discovered_dimension": None,
                    "consensus_score": consensus_score,
                    "decomposed_vector": decomposed_vector,
                    "sparse_active_vector": sparse_active_vector,
                    "confidence": promoted_mod.confidence,
                    "consecutive_anomalies": 0,
                    "events": emitted_events,
                }

        if not is_verified or confirmed_dim_name is None:
            return {
                "status": "FROZEN_IN_H2",
                "prediction": loop_a_result.get("prediction"),
                "observed_outcome": observed_outcome,
                "error": loop_a_result.get("error"),
                "loop_executed": "LOOP_C",
                "loops_executed": loops_executed,
                "active_module_id": None,
                "evaluated_module_id": evaluated_module_id,
                "promoted_module_id": None,
                "discovered_dimension": None,
                "consensus_score": consensus_score,
                "decomposed_vector": decomposed_vector,
                "sparse_active_vector": sparse_active_vector,
                "confidence": current_confidence,
                "consecutive_anomalies": 0,
                "events": emitted_events,
            }

        discovered_dimension = confirmed_dim_name
        if not self.parameter_space.has_dimension(confirmed_dim_name):
            discovered_axis = DimensionAxis(
                name=confirmed_dim_name,
                data_type="continuous",
                variance_threshold=self.sparse_delta_threshold,
                default_value=0.0,
            )
            self.parameter_space.register_dimension(discovered_axis)
            emitted_events.append(
                DimensionDiscoveredEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    discovered_dimension=discovered_axis,
                    triggering_query=target_query,
                )
            )

        if confirmed_dim_name not in current_observation.features.values or (
            current_observation.features.values.get(confirmed_dim_name, 0.0) == 0.0
            and confirmed_dim_name not in raw_input
        ):
            inferred_val = self._extract_inferred_dimension_value(
                external_responses, confirmed_dim_name
            )
            enriched_features = dict(current_observation.features.values)
            enriched_features[confirmed_dim_name] = inferred_val
            current_observation.features = ComponentVector(values=enriched_features)
            for obs in regime_context_window:
                if confirmed_dim_name not in raw_input:
                    ef = dict(obs.features.values)
                    ef[confirmed_dim_name] = inferred_val
                    obs.features = ComponentVector(values=ef)

        # LOOP_D
        loops_executed.append("LOOP_D")
        target_dimensions: Set[str] = (
            set(target_query.active_vector.dimension_keys)
            | set(sparse_active_vector.dimension_keys)
            | {confirmed_dim_name}
        )

        promoted_mod, loop_d_events, loop_d_passed = self._compile_and_promote_module(
            target_query=target_query,
            current_observation=current_observation,
            target_dimensions=target_dimensions,
            regime_observations=regime_context_window if len(regime_context_window) >= 2 else None,
        )
        emitted_events.extend(loop_d_events)

        if loop_d_passed and promoted_mod is not None:
            active_module_id = promoted_mod.module_id
            promoted_module_id = promoted_mod.module_id
            current_confidence = promoted_mod.confidence
            self.router.set_active_context_module(promoted_mod.module_id)
            status = "PROMOTED_TO_H3"
        else:
            active_module_id = None
            promoted_module_id = None
            status = "REGRESSION_FAILED_QUARANTINED"

        return {
            "status": status,
            "prediction": loop_a_result.get("prediction"),
            "observed_outcome": observed_outcome,
            "error": loop_a_result.get("error"),
            "loop_executed": "LOOP_D",
            "loops_executed": loops_executed,
            "active_module_id": active_module_id,
            "evaluated_module_id": evaluated_module_id,
            "promoted_module_id": promoted_module_id,
            "discovered_dimension": discovered_dimension,
            "consensus_score": consensus_score,
            "decomposed_vector": decomposed_vector,
            "sparse_active_vector": sparse_active_vector,
            "confidence": current_confidence,
            "consecutive_anomalies": 0,
            "events": emitted_events,
        }

    def apply_sparse_attention_gate(
        self,
        vector: ComponentVector,
        prior_observations: Optional[List[Observation]] = None,
    ) -> ComponentVector:
        if prior_observations is None:
            prior_observations = self.memory_repo.get_recent_observations(limit=1)

        if not prior_observations:
            return vector

        prev_values = prior_observations[0].features.values
        active_keys: Set[str] = set()

        for key, current_val in vector.values.items():
            if key not in prev_values:
                active_keys.add(key)
            elif abs(current_val - prev_values[key]) > self.sparse_delta_threshold:
                active_keys.add(key)

        if not active_keys:
            return vector

        return vector.extract_subspace(active_keys)

    def _find_matching_h3_specialist(
        self,
        vector: ComponentVector,
        observed_outcome: Any,
        exclude_module_id: Optional[str] = None,
    ) -> Optional[MicroModule]:
        if observed_outcome is None or not isinstance(observed_outcome, (int, float)):
            return None

        best_match: Optional[MicroModule] = None
        best_error: float = float("inf")

        for mod in self.memory_repo.get_active_modules():
            if mod.module_id == exclude_module_id:
                continue
            if not mod.active_dimensions.issubset(vector.dimension_keys):
                continue
            try:
                sub_vec = vector.extract_subspace(mod.active_dimensions)
                pred = mod.evaluate(sub_vec)
                if isinstance(pred, (int, float)) and not isinstance(pred, bool):
                    err = abs(float(observed_outcome) - float(pred))
                    if err <= self.error_threshold and err < best_error:
                        best_error = err
                        best_match = mod
            except CognitiveDomainException:
                continue

        return best_match

    def _compile_and_promote_module(
        self,
        target_query: TargetQuery,
        current_observation: Observation,
        target_dimensions: Set[str],
        regime_observations: Optional[List[Observation]] = None,
    ) -> tuple[Optional[MicroModule], List[DomainEvent], bool]:
        expanded_active_dict: Dict[str, float] = {}
        expanded_delta_dict: Dict[str, float] = dict(target_query.delta_state)

        for dim in target_dimensions:
            default_val = (
                self.parameter_space.get_dimension(dim).default_value
                if self.parameter_space.has_dimension(dim)
                else 0.0
            )
            val = current_observation.features.values.get(
                dim,
                target_query.active_vector.values.get(dim, default_val),
            )
            expanded_active_dict[dim] = val
            if dim not in expanded_delta_dict:
                expanded_delta_dict[dim] = val

        expanded_query = TargetQuery(
            delta_state=expanded_delta_dict,
            active_vector=ComponentVector(values=expanded_active_dict),
            observed_outcome=target_query.observed_outcome,
        )

        source_obs: List[Observation] = (
            list(regime_observations)
            if regime_observations is not None
            else self.memory_repo.get_recent_observations(limit=self.anchor_sample_size)
        )

        projected_context: List[Observation] = []
        for obs in source_obs:
            proj_vals = {
                dim: obs.features.values.get(
                    dim,
                    self.parameter_space.get_dimension(dim).default_value
                    if self.parameter_space.has_dimension(dim)
                    else 0.0,
                )
                for dim in target_dimensions
            }
            projected_context.append(
                Observation(
                    id=obs.id,
                    features=ComponentVector(values=proj_vals),
                    observed_outcome=obs.observed_outcome,
                    timestamp=obs.timestamp,
                )
            )

        candidate_module: MicroModule = self.neural_engine.synthesize_candidate_module(
            target_query=expanded_query,
            context_data=projected_context,
        )
        candidate_module.reinforcement_count = self.anomaly_patience
        self.memory_repo.save_quarantined_module(candidate_module)

        try:
            promoted_event = self.orchestrator.promote_shadow_module(
                candidate=candidate_module,
                anchor_sample_size=self.anchor_sample_size,
                tolerance=self.regression_tolerance,
                anchor_override=projected_context,
            )
            return candidate_module, [promoted_event], True
        except RegressionFailureException:
            candidate_module.attach_exception(
                condition=lambda vec, q_vec=expanded_query.active_vector: (
                    vec.values == q_vec.values
                ),
                outcome=target_query.observed_outcome,
            )
            self.memory_repo.save_quarantined_module(candidate_module)
            return None, [], False

    @staticmethod
    def _extract_inferred_dimension_value(
        external_responses: List[Dict[str, Any]],
        confirmed_dim_name: str,
    ) -> float:
        metrics: List[float] = []
        for item in external_responses:
            if not isinstance(item, dict):
                continue
            dim_val = (
                item.get("dimension_name")
                or item.get("candidate_dimension")
                or item.get("dimension")
                or item.get("name")
            )
            if isinstance(dim_val, str) and dim_val.strip().lower() == confirmed_dim_name.strip().lower():
                for key in ("value", "statistic", "correlation", "effect_size", "score"):
                    raw_m = item.get(key)
                    if isinstance(raw_m, (int, float)) and not isinstance(raw_m, bool):
                        f_m = float(raw_m)
                        if not (math.isnan(f_m) or math.isinf(f_m)) and abs(f_m) > 1e-9:
                            metrics.append(f_m)
                            break

        if not metrics:
            return 1.0

        metrics.sort()
        mid = len(metrics) // 2
        return metrics[mid] if len(metrics) % 2 == 1 else 0.5 * (metrics[mid - 1] + metrics[mid])


class EngineBuilder:
    """
    `CognitiveCoreEngine` nesnesini "Batteries included, fully swappable" felsefesiyle
    ve 5 Builder Kanununa (Kanun #1 - #5) tam uyumlu olarak inşa eden akıcı kurucu sınıf.
    """

    def __init__(self) -> None:
        self._memory_repo: Optional[IMemoryRepository] = None
        self._neural_engine: Optional[INeuralEnginePort] = None
        self._research_agent: Optional[IResearchAgent] = None
        self._consensus_engine: Optional[CrossConsensusEngine] = None
        self._router: Optional[ModuleRouter] = None
        self._parameter_space: Optional[GlobalParameterSpace] = None

        self._initial_dimensions: List[DimensionAxis] = []
        self._initial_active_modules: List[MicroModule] = []
        self._source_providers: List[SourceProvider] = []

        self._error_threshold: float = 0.05
        self._anomaly_patience: int = 3
        self._learning_rate: float = 0.10
        self._decay_rate: float = 0.50
        self._consensus_threshold: float = 0.70
        self._regression_tolerance: float = 0.05
        self._anchor_sample_size: int = 50
        self._sparse_delta_threshold: float = 0.0

    def with_memory_repository(self, repo: IMemoryRepository) -> EngineBuilder:
        if not isinstance(repo, IMemoryRepository):
            raise CognitiveDomainException("with_memory_repository yalnızca IMemoryRepository kabul eder.")
        self._memory_repo = repo
        return self

    def with_neural_engine(self, engine: INeuralEnginePort) -> EngineBuilder:
        if not isinstance(engine, INeuralEnginePort):
            raise CognitiveDomainException("with_neural_engine yalnızca INeuralEnginePort kabul eder.")
        self._neural_engine = engine
        return self

    def with_research_agent(self, agent: IResearchAgent) -> EngineBuilder:
        if not isinstance(agent, IResearchAgent):
            raise CognitiveDomainException("with_research_agent yalnızca IResearchAgent kabul eder.")
        self._research_agent = agent
        return self

    def with_consensus_engine(self, consensus: CrossConsensusEngine) -> EngineBuilder:
        if not isinstance(consensus, CrossConsensusEngine):
            raise CognitiveDomainException("with_consensus_engine yalnızca CrossConsensusEngine kabul eder.")
        self._consensus_engine = consensus
        return self

    def with_router(self, router: ModuleRouter) -> EngineBuilder:
        if not isinstance(router, ModuleRouter):
            raise CognitiveDomainException("with_router yalnızca ModuleRouter kabul eder.")
        self._router = router
        return self

    def with_parameter_space(self, space: GlobalParameterSpace) -> EngineBuilder:
        if not isinstance(space, GlobalParameterSpace):
            raise CognitiveDomainException("with_parameter_space yalnızca GlobalParameterSpace kabul eder.")
        self._parameter_space = space
        return self

    def with_dimension(
        self,
        dimension_or_name: Union[DimensionAxis, str],
        *,
        default_value: float = 0.0,
        variance_threshold: float = 0.0,
        data_type: str = "continuous",
    ) -> EngineBuilder:
        if isinstance(dimension_or_name, DimensionAxis):
            axis = dimension_or_name
        elif isinstance(dimension_or_name, str):
            axis = DimensionAxis(
                name=dimension_or_name.strip(),
                data_type=data_type,
                variance_threshold=variance_threshold,
                default_value=default_value,
            )
        else:
            raise CognitiveDomainException("with_dimension bir DimensionAxis veya str olmalıdır.")

        if any(d.name == axis.name for d in self._initial_dimensions):
            raise EpistemicInvariantViolationException(
                f"Boyut çakışması: '{axis.name}' zaten tanımlanmış.",
                context={"dimension_name": axis.name},
            )

        self._initial_dimensions.append(axis)
        return self

    def with_initial_module(self, module: MicroModule) -> EngineBuilder:
        if not isinstance(module, MicroModule):
            raise CognitiveDomainException("with_initial_module yalnızca MicroModule kabul eder.")
        self._initial_active_modules.append(module)
        return self

    def with_source_provider(self, provider: SourceProvider) -> EngineBuilder:
        if not isinstance(provider, SourceProvider):
            raise CognitiveDomainException("with_source_provider yalnızca SourceProvider kabul eder.")
        existing_ids = {p.source_id for p in self._source_providers}
        if provider.source_id in existing_ids:
            raise EpistemicInvariantViolationException(
                f"Kaynak çakışması: '{provider.source_id}' zaten eklendi.",
                context={"source_id": provider.source_id},
            )
        self._source_providers.append(provider)
        return self

    def with_hyperparameters(
        self,
        *,
        error_threshold: Optional[float] = None,
        anomaly_patience: Optional[int] = None,
        learning_rate: Optional[float] = None,
        decay_rate: Optional[float] = None,
        consensus_threshold: Optional[float] = None,
        regression_tolerance: Optional[float] = None,
        anchor_sample_size: Optional[int] = None,
        sparse_delta_threshold: Optional[float] = None,
    ) -> EngineBuilder:
        if error_threshold is not None:
            if error_threshold < 0.0 or math.isnan(error_threshold):
                raise CognitiveDomainException("error_threshold negatif olamaz.")
            self._error_threshold = float(error_threshold)

        if anomaly_patience is not None:
            if anomaly_patience <= 0 or isinstance(anomaly_patience, bool):
                raise CognitiveDomainException("anomaly_patience pozitif olmalıdır.")
            self._anomaly_patience = int(anomaly_patience)

        if learning_rate is not None:
            if not (0.0 < learning_rate < 1.0):
                raise EpistemicInvariantViolationException("learning_rate (0.0, 1.0) aralığında olmalıdır.")
            self._learning_rate = float(learning_rate)

        if decay_rate is not None:
            if not (0.0 < decay_rate < 1.0):
                raise EpistemicInvariantViolationException("decay_rate (0.0, 1.0) aralığında olmalıdır.")
            self._decay_rate = float(decay_rate)

        if consensus_threshold is not None:
            if not (0.0 < consensus_threshold <= 1.0):
                raise EpistemicInvariantViolationException("consensus_threshold (0.0, 1.0] aralığında olmalıdır.")
            self._consensus_threshold = float(consensus_threshold)

        if regression_tolerance is not None:
            if regression_tolerance < 0.0 or math.isnan(regression_tolerance):
                raise CognitiveDomainException("regression_tolerance negatif olamaz.")
            self._regression_tolerance = float(regression_tolerance)

        if anchor_sample_size is not None:
            if anchor_sample_size <= 0:
                raise CognitiveDomainException("anchor_sample_size pozitif olmalıdır.")
            self._anchor_sample_size = int(anchor_sample_size)

        if sparse_delta_threshold is not None:
            if sparse_delta_threshold < 0.0 or math.isnan(sparse_delta_threshold):
                raise CognitiveDomainException("sparse_delta_threshold negatif olamaz.")
            self._sparse_delta_threshold = float(sparse_delta_threshold)

        return self

    def build(self) -> CognitiveCoreEngine:
        memory_repo = (
            self._memory_repo if self._memory_repo is not None else InMemoryMemoryRepository()
        )
        neural_engine = (
            self._neural_engine
            if self._neural_engine is not None
            else PurePythonNeuralEngine(training_epochs=1500, step_size=0.60)
        )
        consensus_engine = (
            self._consensus_engine
            if self._consensus_engine is not None
            else CrossConsensusEngine(min_independent_sources=2)
        )
        parameter_space = (
            self._parameter_space if self._parameter_space is not None else GlobalParameterSpace()
        )
        router = (
            self._router
            if self._router is not None
            else ModuleRouter(memory_repo=memory_repo, similarity_metric="euclidean")
        )
        if router.memory_repo is None:
            router.memory_repo = memory_repo

        if self._research_agent is None:
            research_agent = MockMultiSourceResearchAgent(providers=list(self._source_providers))
        else:
            research_agent = self._research_agent
            if self._source_providers and isinstance(research_agent, MockMultiSourceResearchAgent):
                existing_ids = {p.source_id for p in research_agent.providers}
                for provider in self._source_providers:
                    if provider.source_id not in existing_ids:
                        research_agent.register_provider(provider)

        for dim in self._initial_dimensions:
            if not parameter_space.has_dimension(dim.name):
                parameter_space.register_dimension(dim)

        for mod in self._initial_active_modules:
            for dim_name in mod.active_dimensions:
                if not parameter_space.has_dimension(dim_name):
                    parameter_space.register_dimension(
                        DimensionAxis(
                            name=dim_name,
                            data_type="continuous",
                            variance_threshold=self._sparse_delta_threshold,
                            default_value=0.0,
                        )
                    )
            memory_repo.save_active_module(mod)
            router.register_module(mod)

        return CognitiveCoreEngine(
            memory_repo=memory_repo,
            neural_engine=neural_engine,
            research_agent=research_agent,
            consensus_engine=consensus_engine,
            parameter_space=parameter_space,
            router=router,
            error_threshold=self._error_threshold,
            anomaly_patience=self._anomaly_patience,
            learning_rate=self._learning_rate,
            decay_rate=self._decay_rate,
            consensus_threshold=self._consensus_threshold,
            regression_tolerance=self._regression_tolerance,
            anchor_sample_size=self._anchor_sample_size,
            sparse_delta_threshold=self._sparse_delta_threshold,
        )


Engine = CognitiveCoreEngine

__all__ = [
    "CognitiveCoreEngine",
    "Engine",
    "EngineBuilder",
    "ModuleRouter",
]