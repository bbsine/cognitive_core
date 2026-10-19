"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Sprint 2 Application Orkestrasyon Test Süiti.

Bu test modülü; dış kütüphane veya 'unittest.mock' kullanmadan, SPI portlarını
gerçekleyen saf Python test dublörleri (InMemoryRepository ve StubNeuralEnginePort)
üzerinden CognitiveOrchestrator servisinin Döngü A, Döngü B ve Döngü D
davranışlarını doğrular.

Çalıştırma komutu:
    python -m unittest tests/unit/test_application_orchestrator.py -v
"""

from __future__ import annotations
from typing import Any, Dict, List
import unittest

from core.application import (
    CognitiveOrchestrator,
    IMemoryRepository,
    INeuralEnginePort,
)
from core.domain import (
    AnomalyQuarantinedEvent,
    ComponentVector,
    ConfidenceScore,
    DimensionAxis,
    GlobalParameterSpace,
    MicroModule,
    ModuleDemotedEvent,
    ModulePromotedEvent,
    Observation,
    RegressionFailureException,
    TargetQuery,
)


# =============================================================================
# SAF PYTHON TEST DUBLÖRLERİ (IN-MEMORY & STUB SPI IMPLEMENTASYONLARI)
# =============================================================================


class InMemoryRepository(IMemoryRepository):
    """
    H_1, H_2 ve H_3 epistemik havuzlarını bellekte (RAM) simüle eden
    saf Python IMemoryRepository test dublörü.
    """

    def __init__(self) -> None:
        self._h1_observations: List[Observation] = []
        self._h2_quarantine: Dict[str, MicroModule] = {}
        self._h3_active: Dict[str, MicroModule] = {}

    def save_observation(self, observation: Observation) -> None:
        self._h1_observations.append(observation)

    def get_recent_observations(self, limit: int) -> List[Observation]:
        if limit <= 0:
            return []
        # En güncel gözlemden eskiye doğru sıralı döndür
        return list(reversed(self._h1_observations))[:limit]

    def save_quarantined_module(self, module: MicroModule) -> None:
        self._h2_quarantine[module.module_id] = module

    def get_quarantined_modules(self) -> List[MicroModule]:
        return list(self._h2_quarantine.values())

    def save_active_module(self, module: MicroModule) -> None:
        self._h3_active[module.module_id] = module

    def get_active_modules(self) -> List[MicroModule]:
        return list(self._h3_active.values())

    def remove_quarantined_module(self, module_id: str) -> None:
        self._h2_quarantine.pop(module_id, None)

    def remove_active_module(self, module_id: str) -> None:
        self._h3_active.pop(module_id, None)


class StubNeuralEnginePort(INeuralEnginePort):
    """
    Nitelik ayrıştırma Φ(X), aday modül sentezi ve Döngü D regresyon testini
    simüle eden yapılandırılabilir INeuralEnginePort test dublörü.
    """

    def __init__(self, regression_pass: bool = True) -> None:
        self.regression_pass: bool = regression_pass
        self.last_evaluated_anchor_count: int = 0
        self.last_evaluated_tolerance: float = 0.0

    def decompose_input(self, raw_input: Dict[str, Any]) -> ComponentVector:
        numeric_features: Dict[str, float] = {
            k: float(v)
            for k, v in raw_input.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        return ComponentVector(values=numeric_features)

    def synthesize_candidate_module(
        self,
        target_query: TargetQuery,
        context_data: List[Observation],
    ) -> MicroModule:
        return MicroModule(
            module_id="synthesized_m_new",
            active_dimensions=target_query.active_vector.dimension_keys,
            confidence=ConfidenceScore(score=0.5),
            predict_fn=lambda vec: sum(vec.values.values()),
        )

    def evaluate_regression(
        self,
        candidate: MicroModule,
        anchor_data: List[Observation],
        tolerance: float,
    ) -> bool:
        self.last_evaluated_anchor_count = len(anchor_data)
        self.last_evaluated_tolerance = tolerance
        return self.regression_pass


# =============================================================================
# SPRINT 2 ORKESTRASYON BİRİM TESTLERİ
# =============================================================================


class TestCognitiveOrchestrator(unittest.TestCase):
    """
    CognitiveOrchestrator (Döngü A, Döngü B ve Döngü D) Senaryo Testleri.
    """

    def setUp(self) -> None:
        self.memory_repo = InMemoryRepository()
        self.neural_engine = StubNeuralEnginePort(regression_pass=True)
        self.parameter_space = GlobalParameterSpace()

        # Temel boyut eksenlerini kaydet
        self.parameter_space.register_dimension(
            DimensionAxis(name="v_1", data_type="continuous", variance_threshold=0.1)
        )
        self.parameter_space.register_dimension(
            DimensionAxis(name="v_2", data_type="continuous", variance_threshold=0.1)
        )

        self.orchestrator = CognitiveOrchestrator(
            memory_repo=self.memory_repo,
            neural_engine=self.neural_engine,
            parameter_space=self.parameter_space,
        )

    def test_loop_a_successful_inference_and_asymptotic_reinforcement(self) -> None:
        """
        Senaryo 1 (Döngü A - Pekiştirme):
        |Y_obs - Y_pred| <= error_threshold olduğunda aktif modülün güven skoru
        asimptotik olarak artmalı, H_3'te güncellenmeli ve durum 'REINFORCED' dönmelidir.
        """
        initial_score = 0.60
        learning_rate = 0.20
        module = MicroModule(
            module_id="mod_linear_v1",
            active_dimensions={"v_1"},
            confidence=ConfidenceScore(score=initial_score),
            predict_fn=lambda vec: vec.values["v_1"] * 2.0,
        )
        self.memory_repo.save_active_module(module)

        # Girdi: v_1 = 5.0 -> Tahmin (Y_pred) = 10.0, Gözlem (Y_obs) = 10.03
        # Hata = |10.03 - 10.0| = 0.03 <= 0.05 (Eşik)
        result = self.orchestrator.infer_and_adapt(
            raw_input={"v_1": 5.0, "v_2": 100.0},
            observed_outcome=10.03,
            error_threshold=0.05,
            learning_rate=learning_rate,
            decay_rate=0.5,
        )

        expected_confidence = initial_score + learning_rate * (
            ConfidenceScore.DEFAULT_P_MAX - initial_score
        )

        self.assertEqual(result["status"], "REINFORCED")
        self.assertEqual(result["module_id"], "mod_linear_v1")
        self.assertAlmostEqual(result["prediction"], 10.0, places=9)
        self.assertAlmostEqual(result["error"], 0.03, places=9)
        self.assertAlmostEqual(result["confidence"].score, expected_confidence, places=12)
        self.assertIsNone(result["target_query"])
        self.assertEqual(len(result["events"]), 0)

        # Seyrek dikkat kapısı (G_sparse) kontrolü: V_aktif yalnızca {'v_1'} içermeli
        self.assertEqual(result["active_vector"].dimension_keys, {"v_1"})

        # H_3 güncellenmiş olmalı, H_2 boş kalmalı, H_1'de 1 gözlem bulunmalı
        active_modules = self.memory_repo.get_active_modules()
        self.assertEqual(len(active_modules), 1)
        self.assertAlmostEqual(active_modules[0].confidence.score, expected_confidence, places=12)
        self.assertEqual(len(self.memory_repo.get_quarantined_modules()), 0)
        self.assertEqual(len(self.memory_repo.get_recent_observations(limit=10)), 1)

    def test_loop_a_anomaly_detection_and_demotion_to_quarantine(self) -> None:
        """
        Senaryo 2 (Döngü A - Anomali ve Karantina):
        |Y_obs - Y_pred| > error_threshold olduğunda modülün güveni γ çarpanı ile
        düşürülmeli, modül H_3'ten silinip H_2'ye taşınmalı, TargetQuery ve
        ilgili domain olayları (ModuleDemotedEvent, AnomalyQuarantinedEvent) üretilmelidir.
        """
        initial_score = 0.80
        decay_rate = 0.50
        module = MicroModule(
            module_id="mod_fragile",
            active_dimensions={"v_1"},
            confidence=ConfidenceScore(score=initial_score),
            predict_fn=lambda vec: vec.values["v_1"] * 2.0,
        )
        self.memory_repo.save_active_module(module)

        # Delta (Δv_1) hesabını doğrulamak için önce H_1'e referans bir gözlem ekle
        self.orchestrator.infer_and_adapt(
            raw_input={"v_1": 3.0, "v_2": 1.0},
            observed_outcome=6.0,
            error_threshold=0.05,
            learning_rate=0.1,
            decay_rate=decay_rate,
        )
        reinforced_score = module.confidence.score

        # Şimdi anomali yaratan gözlemi besle: v_1 = 5.0 -> Y_pred = 10.0, fakat Y_obs = 25.0
        # Hata = |25.0 - 10.0| = 15.0 > 0.05
        anomaly_result = self.orchestrator.infer_and_adapt(
            raw_input={"v_1": 5.0, "v_2": 1.0},
            observed_outcome=25.0,
            error_threshold=0.05,
            learning_rate=0.1,
            decay_rate=decay_rate,
        )

        expected_penalized_score = reinforced_score * decay_rate

        self.assertEqual(anomaly_result["status"], "ANOMALY_QUARANTINED")
        self.assertEqual(anomaly_result["module_id"], "mod_fragile")
        self.assertAlmostEqual(anomaly_result["error"], 15.0, places=9)
        self.assertAlmostEqual(
            anomaly_result["confidence"].score, expected_penalized_score, places=12
        )

        # Havuz geçiş doğrulaması: H_3 boşalmalı, modül H_2'ye geçmeli
        self.assertEqual(len(self.memory_repo.get_active_modules()), 0)
        quarantined = self.memory_repo.get_quarantined_modules()
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0].module_id, "mod_fragile")

        # TargetQuery Q(Δ, V_aktif, Y_obs) doğrulaması (Δv_1 = 5.0 - 3.0 = 2.0)
        target_query = anomaly_result["target_query"]
        self.assertIsInstance(target_query, TargetQuery)
        self.assertAlmostEqual(target_query.delta_state["v_1"], 2.0, places=9)
        self.assertEqual(target_query.observed_outcome, 25.0)

        # Domain Events doğrulaması
        events = anomaly_result["events"]
        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], ModuleDemotedEvent)
        self.assertIsInstance(events[1], AnomalyQuarantinedEvent)
        self.assertEqual(events[0].source_pool, "H_3")
        self.assertEqual(events[0].target_pool, "H_2")

    def test_loop_b_internal_diagnosis_escalation_and_internal_explanation(self) -> None:
        """
        Senaryo 3 (Döngü B - İç Uzay Taraması ve Katman 4 Eskalasyonu):
        - Pasif boyutlarda varyans eşiği aşılmadığında 'ESCALATE_TO_EXTERNAL' dönmelidir.
        - Pasif boyutta varyans eşiğini aşan değişim varsa 'EXPLAINED_INTERNALLY' dönmelidir.
        """
        # Varyans eşiği 0.5 olan pasif bir boyut (theta_latent) tanımla
        self.parameter_space.register_dimension(
            DimensionAxis(name="theta_latent", data_type="continuous", variance_threshold=0.5)
        )

        active_vec = ComponentVector(values={"v_1": 5.0})
        query = TargetQuery(
            delta_state={"v_1": 2.0},
            active_vector=active_vec,
            observed_outcome=25.0,
        )

        # 3a. Alt Senaryo: H_1'deki gözlemlerde pasif boyutlar (v_2, theta_latent) sabit (Varyans = 0.0)
        self.memory_repo.save_observation(
            Observation(
                id="obs_1",
                features=ComponentVector(values={"v_1": 3.0, "v_2": 1.0, "theta_latent": 2.0}),
                observed_outcome=6.0,
            )
        )
        self.memory_repo.save_observation(
            Observation(
                id="obs_2",
                features=ComponentVector(values={"v_1": 5.0, "v_2": 1.0, "theta_latent": 2.1}),
                observed_outcome=25.0,
            )
        )

        diag_escalated = self.orchestrator.diagnose_internal_space(target_query=query)
        self.assertEqual(diag_escalated["status"], "ESCALATE_TO_EXTERNAL")
        self.assertFalse(diag_escalated["explained_internally"])
        self.assertEqual(diag_escalated["passive_dimensions"], {"v_2", "theta_latent"})
        self.assertEqual(diag_escalated["candidate_dimensions"], set())

        # 3b. Alt Senaryo: H_1'e theta_latent boyutunda yüksek varyanslı (2.0 -> 10.0, fark=8.0 > 0.5) gözlem ekle
        self.memory_repo.save_observation(
            Observation(
                id="obs_3",
                features=ComponentVector(values={"v_1": 5.5, "v_2": 1.0, "theta_latent": 10.0}),
                observed_outcome=40.0,
            )
        )

        diag_explained = self.orchestrator.diagnose_internal_space(target_query=query)
        self.assertEqual(diag_explained["status"], "EXPLAINED_INTERNALLY")
        self.assertTrue(diag_explained["explained_internally"])
        self.assertEqual(diag_explained["candidate_dimensions"], {"theta_latent"})

    def test_loop_d_regression_test_success_promotes_module_to_h3(self) -> None:
        """
        Senaryo 4a (Döngü D - Başarılı Regresyon ve Terfi):
        evaluate_regression True döndüğünde aday modül H_2'den silinmeli,
        H_3 aktif çekirdeğine kaydedilmeli ve ModulePromotedEvent dönmelidir.
        """
        self.neural_engine.regression_pass = True

        # H_1'e çapa (anchor) verileri ekle
        for i in range(5):
            self.memory_repo.save_observation(
                Observation(
                    id=f"anchor_{i}",
                    features=ComponentVector(values={"v_1": float(i), "v_2": 1.0}),
                    observed_outcome=float(i * 2),
                )
            )

        candidate_module = MicroModule(
            module_id="mod_candidate_v1_v2",
            active_dimensions={"v_1", "v_2"},
            confidence=ConfidenceScore(score=0.55),
            predict_fn=lambda vec: vec.values["v_1"] * 2.0 + vec.values["v_2"],
        )
        self.memory_repo.save_quarantined_module(candidate_module)

        promoted_event = self.orchestrator.promote_shadow_module(
            candidate=candidate_module,
            anchor_sample_size=5,
            tolerance=0.02,
        )

        self.assertIsInstance(promoted_event, ModulePromotedEvent)
        self.assertEqual(promoted_event.module_id, "mod_candidate_v1_v2")
        self.assertEqual(promoted_event.source_pool, "H_2")
        self.assertEqual(promoted_event.target_pool, "H_3")

        # H_2'den silinmiş ve H_3'e eklenmiş olmalı
        self.assertEqual(len(self.memory_repo.get_quarantined_modules()), 0)
        active_modules = self.memory_repo.get_active_modules()
        self.assertEqual(len(active_modules), 1)
        self.assertEqual(active_modules[0].module_id, "mod_candidate_v1_v2")
        self.assertEqual(self.neural_engine.last_evaluated_anchor_count, 5)
        self.assertAlmostEqual(self.neural_engine.last_evaluated_tolerance, 0.02)

    def test_loop_d_regression_test_failure_raises_exception_and_keeps_in_h2(self) -> None:
        """
        Senaryo 4b (Döngü D - Catastrophic Forgetting Koruması / Invariant #2):
        evaluate_regression False döndüğünde RegressionFailureException fırlatılmalı,
        aday modül H_3'e kesinlikle geçmemeli ve H_2 karantinasında kalmalıdır.
        """
        self.neural_engine.regression_pass = False

        candidate_module = MicroModule(
            module_id="mod_destructive",
            active_dimensions={"v_1"},
            confidence=ConfidenceScore(score=0.50),
            predict_fn=lambda vec: -999.0,
        )
        self.memory_repo.save_quarantined_module(candidate_module)

        with self.assertRaises(RegressionFailureException):
            self.orchestrator.promote_shadow_module(
                candidate=candidate_module,
                anchor_sample_size=20,
                tolerance=0.05,
            )

        # Invariant #2 doğrulaması: H_3 boş kalmalı, modül H_2'de korunmalı
        self.assertEqual(len(self.memory_repo.get_active_modules()), 0)
        quarantined = self.memory_repo.get_quarantined_modules()
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0].module_id, "mod_destructive")


if __name__ == "__main__":
    unittest.main()