"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Sprint 5 SDK Facade & EngineBuilder Kanunları Test Süiti.

Bu test modülü; 'core.interfaces' paketindeki EngineBuilder, ModuleRouter ve
CognitiveCoreEngine (Facade) sınıflarını:
- EngineBuilder'ın 5 Mimari Kanunu (Batteries-Included Default, Domain-Agnostic
  Parameter Space, Swappable Oracle Injection, Fluent API & Pre-flight Validation,
  Çift Yönlü Modül ve Uzay Senkronizasyonu)
- Döngü A, Döngü B, Döngü C ve Döngü D uçtan uca bilişsel akışları
üzerinden doğrular.

Çalıştırma komutu:
    python -m unittest tests/unit/test_engine_facade.py -v
"""

from __future__ import annotations
import unittest

from core.domain import (
    AnomalyQuarantinedEvent,
    CognitiveDomainException,
    ComponentVector,
    ConfidenceScore,
    DimensionAxis,
    DimensionDiscoveredEvent,
    EpistemicInvariantViolationException,
    MicroModule,
    ModuleDemotedEvent,
    ModulePromotedEvent,
    TargetQuery,
)
from core.infrastructure import (
    InMemoryMemoryRepository,
    PurePythonNeuralEngine,
)
from core.infrastructure.research_agent import (
    CrossConsensusEngine,
    MockMultiSourceResearchAgent,
    SourceProvider,
)
from core.interfaces import (
    CognitiveCoreEngine,
    EngineBuilder,
    ModuleRouter,
)


class TestEngineBuilderLaws(unittest.TestCase):
    """
    EngineBuilder Mimari Şartnamesindeki 5 Temel Kanunun (Kanun #1 - #5) Doğrulanması.
    """

    def test_law_1_zero_config_batteries_included_defaults(self) -> None:
        """
        Kanun #1 (Sıfır Konfigürasyon ile Kararlı Başlangıç):
        Hiçbir altyapı nesnesi verilmeden EngineBuilder().build() çağrıldığında
        sistem deterministik varsayılanlarla ayağa kalkmalıdır.
        """
        engine = EngineBuilder().build()

        self.assertIsInstance(engine, CognitiveCoreEngine)
        self.assertIsInstance(engine.memory_repo, InMemoryMemoryRepository)
        self.assertIsInstance(engine.neural_engine, PurePythonNeuralEngine)
        self.assertEqual(engine.neural_engine.training_epochs, 1500)
        self.assertAlmostEqual(engine.neural_engine.step_size, 0.60)
        self.assertIsInstance(engine.router, ModuleRouter)
        self.assertEqual(engine.router.similarity_metric, "euclidean")
        self.assertIs(engine.router.memory_repo, engine.memory_repo)
        self.assertIsInstance(engine.consensus_engine, CrossConsensusEngine)
        self.assertEqual(engine.consensus_engine.min_independent_sources, 2)
        self.assertEqual(engine.anomaly_patience, 3)

    def test_law_2_domain_agnostic_dimension_registration_overloads(self) -> None:
        """
        Kanun #2 (Mutlak Alan Bağımsızlığı & Esnek Boyut Kaydı):
        .with_dimension() hem DimensionAxis nesnesi hem de pratik argümanlar
        (name, default_value, variance_threshold, data_type) kabul etmelidir.
        """
        engine = (
            EngineBuilder()
            .with_dimension(DimensionAxis(name="dim_axis_obj", data_type="continuous", variance_threshold=0.02))
            .with_dimension(
                "dim_from_kwargs",
                default_value=1.5,
                variance_threshold=0.08,
                data_type="continuous",
            )
            .build()
        )

        self.assertTrue(engine.parameter_space.has_dimension("dim_axis_obj"))
        self.assertTrue(engine.parameter_space.has_dimension("dim_from_kwargs"))
        self.assertAlmostEqual(engine.dimension_defaults["dim_axis_obj"], 0.0)
        self.assertAlmostEqual(engine.dimension_defaults["dim_from_kwargs"], 1.5)
        self.assertAlmostEqual(
            engine.parameter_space.get_dimension("dim_from_kwargs").variance_threshold,
            0.08,
        )

    def test_law_3_polymorphic_oracle_injection_channels(self) -> None:
        """
        Kanun #3 (Polimorfik Doğrulama Ajanı):
        1. Doğrudan IResearchAgent enjeksiyonu (.with_research_agent)
        2. Hafif SourceProvider zincirlemesi (.with_source_provider -> otomatik MockMultiSourceResearchAgent)
        desteklenmelidir.
        """
        # Kanal 2: Hafif SourceProvider zinciri
        p1 = SourceProvider(
            source_id="oracle_1",
            trust_weight=1.0,
            query_fn=lambda q: {"dimension_name": "latent_x", "statistic": 2.0},
        )
        p2 = SourceProvider(
            source_id="oracle_2",
            trust_weight=1.5,
            static_payload={"dimension_name": "latent_x", "statistic": 2.0},
        )
        engine_with_providers = (
            EngineBuilder()
            .with_source_provider(p1)
            .with_source_provider(p2)
            .build()
        )
        self.assertIsInstance(engine_with_providers.research_agent, MockMultiSourceResearchAgent)
        query = TargetQuery(
            delta_state={"v_1": 1.0},
            active_vector=ComponentVector(values={"v_1": 1.0}),
            observed_outcome=4.0,
        )
        responses = engine_with_providers.research_agent.query_external_sources(query)
        self.assertEqual(len(responses), 2)

        # Kanal 1: Tam IResearchAgent enjeksiyonu
        custom_agent = MockMultiSourceResearchAgent(providers=[p1])
        engine_with_custom_agent = (
            EngineBuilder()
            .with_research_agent(custom_agent)
            .build()
        )
        self.assertIs(engine_with_custom_agent.research_agent, custom_agent)

    def test_law_4_fluent_api_and_preflight_validation_shield(self) -> None:
        """
        Kanun #4 (Akıcı Arayüz ve Doğrulama Kalkanı):
        - Boyut isim çakışmaları EpistemicInvariantViolationException fırlatmalıdır.
        - Geçersiz hiperparametreler (P >= 1.0, negatif tolerans, anomaly_patience <= 0)
          anında reddedilmelidir.
        """
        # Boyut çakışması testi
        builder = EngineBuilder().with_dimension("dim_dup", default_value=0.0)
        with self.assertRaises(EpistemicInvariantViolationException):
            builder.with_dimension("dim_dup", default_value=1.0)

        # learning_rate >= 1.0 veya <= 0.0 (Invariant #1 ihlali)
        with self.assertRaises(EpistemicInvariantViolationException):
            EngineBuilder().with_hyperparameters(learning_rate=1.0)

        with self.assertRaises(EpistemicInvariantViolationException):
            EngineBuilder().with_hyperparameters(decay_rate=0.0)

        with self.assertRaises(EpistemicInvariantViolationException):
            EngineBuilder().with_hyperparameters(consensus_threshold=1.2)

        # anomaly_patience <= 0
        with self.assertRaises(CognitiveDomainException):
            EngineBuilder().with_hyperparameters(anomaly_patience=0)

        # error_threshold < 0.0
        with self.assertRaises(CognitiveDomainException):
            EngineBuilder().with_hyperparameters(error_threshold=-0.1)

    def test_law_5_bidirectional_module_and_space_synchronization(self) -> None:
        """
        Kanun #5 (Çift Yönlü Modül ve Uzay Senkronizasyonu):
        .with_initial_module(module) çağrıldığında:
        - Modülün kullandığı ve henüz uzayda olmayan boyutlar default_value=0.0 ile
          GlobalParameterSpace'e otomatik kaydedilmelidir.
        - Modül hem IMemoryRepository (H_3) hem de ModuleRouter kütüphanesine
          otomatik tescil edilmelidir.
        """
        seed_module = MicroModule(
            module_id="mod_auto_sync",
            active_dimensions={"auto_dim_1", "auto_dim_2"},
            confidence=ConfidenceScore(score=0.75),
            predict_fn=lambda vec: vec.values["auto_dim_1"] + vec.values["auto_dim_2"],
        )

        custom_router = ModuleRouter(similarity_metric="cosine")
        engine = (
            EngineBuilder()
            .with_router(custom_router)
            .with_initial_module(seed_module)
            .build()
        )

        # Boyutlar GlobalParameterSpace'e default_value=0.0 ile otomatik eklenmiş olmalı
        self.assertTrue(engine.parameter_space.has_dimension("auto_dim_1"))
        self.assertTrue(engine.parameter_space.has_dimension("auto_dim_2"))
        self.assertAlmostEqual(engine.dimension_defaults["auto_dim_1"], 0.0)
        self.assertAlmostEqual(engine.dimension_defaults["auto_dim_2"], 0.0)

        # Modül hem IMemoryRepository hem de ModuleRouter içinde tescilli olmalı
        repo_ids = {m.module_id for m in engine.memory_repo.get_active_modules()}
        router_ids = {m.module_id for m in engine.router.get_registered_modules()}
        self.assertIn("mod_auto_sync", repo_ids)
        self.assertIn("mod_auto_sync", router_ids)

        routed = engine.router.route(
            ComponentVector(values={"auto_dim_1": 1.0, "auto_dim_2": 2.0})
        )
        self.assertIsNotNone(routed)
        assert routed is not None
        self.assertEqual(routed.module_id, "mod_auto_sync")


class TestCognitiveCoreEngineFacade(unittest.TestCase):
    """
    EngineBuilder ve CognitiveCoreEngine Uçtan Uca Bilişsel Döngü (A, B, C, D) Testleri.
    """

    def test_engine_builder_fluent_configuration_and_initialization(self) -> None:
        """
        Senaryo 1 (EngineBuilder Doğrulaması):
        Akıcı (fluent) API üzerinden tanımlanan boyutlar, başlangıç modülleri,
        dış veri kaynakları ve hiperparametreler CognitiveCoreEngine örneğine
        eksiksiz enjekte edilmelidir.
        """
        seed_module = MicroModule(
            module_id="mod_seed_v1",
            active_dimensions={"v_1"},
            confidence=ConfidenceScore(score=0.65),
            predict_fn=lambda vec: vec.values["v_1"] * 2.0,
        )
        provider_1 = SourceProvider(
            source_id="src_alpha",
            trust_weight=1.0,
            static_payload={"dimension_name": "theta_ext", "statistic": 2.0},
        )
        provider_2 = SourceProvider(
            source_id="src_beta",
            trust_weight=1.2,
            static_payload={"dimension_name": "theta_ext", "statistic": 2.0},
        )

        engine = (
            EngineBuilder()
            .with_dimension(DimensionAxis(name="v_1", data_type="continuous", variance_threshold=0.05))
            .with_dimension("v_2", default_value=0.0, variance_threshold=0.10, data_type="continuous")
            .with_initial_module(seed_module)
            .with_source_provider(provider_1)
            .with_source_provider(provider_2)
            .with_hyperparameters(
                error_threshold=0.04,
                anomaly_patience=4,
                learning_rate=0.15,
                decay_rate=0.45,
                consensus_threshold=0.75,
                regression_tolerance=0.02,
                anchor_sample_size=25,
                sparse_delta_threshold=0.01,
            )
            .build()
        )

        self.assertIsInstance(engine, CognitiveCoreEngine)
        self.assertTrue(engine.parameter_space.has_dimension("v_1"))
        self.assertTrue(engine.parameter_space.has_dimension("v_2"))

        active_modules = engine.memory_repo.get_active_modules()
        self.assertEqual(len(active_modules), 1)
        self.assertEqual(active_modules[0].module_id, "mod_seed_v1")

        self.assertAlmostEqual(engine.error_threshold, 0.04)
        self.assertEqual(engine.anomaly_patience, 4)
        self.assertAlmostEqual(engine.learning_rate, 0.15)
        self.assertAlmostEqual(engine.decay_rate, 0.45)
        self.assertAlmostEqual(engine.consensus_threshold, 0.75)
        self.assertAlmostEqual(engine.regression_tolerance, 0.02)
        self.assertEqual(engine.anchor_sample_size, 25)
        self.assertAlmostEqual(engine.sparse_delta_threshold, 0.01)

    def test_full_loop_a_flow_direct_prediction_and_reinforcement(self) -> None:
        """
        Senaryo 2 (Tam Döngü A Akışı - Pekiştirme):
        H_3'teki aktif modül hata eşiği dahilinde (|Y_obs - Y_pred| <= ε) tahmin
        ürettiğinde adım durumu 'REINFORCED' olmalı, yalnızca 'LOOP_A' çalışmalı
        ve modülün güven skoru asimptotik olarak artmalıdır.
        """
        initial_conf = 0.60
        learning_rate = 0.20
        seed_module = MicroModule(
            module_id="mod_linear_v1",
            active_dimensions={"v_1"},
            confidence=ConfidenceScore(score=initial_conf),
            predict_fn=lambda vec: vec.values["v_1"] * 3.0,
        )

        engine = (
            EngineBuilder()
            .with_initial_module(seed_module)
            .with_hyperparameters(
                error_threshold=0.05,
                learning_rate=learning_rate,
            )
            .build()
        )

        telemetry = engine.step(
            raw_input={"v_1": 4.0, "noise_flag": True},
            observed_outcome=12.02,
        )

        expected_conf = initial_conf + learning_rate * (
            ConfidenceScore.DEFAULT_P_MAX - initial_conf
        )

        self.assertEqual(telemetry["status"], "REINFORCED")
        self.assertEqual(telemetry["loop_executed"], "LOOP_A")
        self.assertEqual(telemetry["loops_executed"], ["LOOP_A"])
        self.assertEqual(telemetry["active_module_id"], "mod_linear_v1")
        self.assertAlmostEqual(telemetry["prediction"], 12.0, places=9)
        self.assertAlmostEqual(telemetry["error"], 0.02, places=9)
        self.assertAlmostEqual(telemetry["confidence"].score, expected_conf, places=12)
        self.assertLess(telemetry["confidence"].score, 1.0)
        self.assertEqual(len(telemetry["events"]), 0)

    def test_full_loop_b_flow_internal_space_explanation_and_promotion(self) -> None:
        """
        Senaryo 3 (Tam Döngü B Akışı - İç Uzayda Çözüm ve Alt-Ağ Derleme):
        Aktif modül yalnız 'v_1' boyutunu kullanırken bir anomali oluştuğunda,
        eğer Ω_global içinde zaten kayıtlı olan pasif 'v_2' boyutu bu adımda
        varyans eşiğini aşan bir değişim göstermişse:
        1. Döngü A modülü karantinaya almalı,
        2. Döngü B anomaliyi iç uzayda açıklamalı ('v_2' aday boyutunu bularak),
        3. Döngü D {'v_1', 'v_2'} boyutlu yeni modülü sentezleyip regresyon testinden
           geçirerek H_3'e terfi ettirmeli ve durum 'EXPLAINED_INTERNALLY' dönmelidir.
        """
        seed_module = MicroModule(
            module_id="mod_only_v1",
            active_dimensions={"v_1"},
            confidence=ConfidenceScore(score=0.75),
            predict_fn=lambda vec: vec.values["v_1"] * 2.0,
        )

        engine = (
            EngineBuilder()
            .with_dimension("v_1", default_value=0.0, variance_threshold=0.05)
            .with_dimension("v_2", default_value=0.0, variance_threshold=0.10)
            .with_initial_module(seed_module)
            .with_hyperparameters(
                error_threshold=0.05,
                regression_tolerance=0.05,
            )
            .build()
        )

        step_1 = engine.step(
            raw_input={"v_1": 2.0, "v_2": 0.0},
            observed_outcome=4.0,
        )
        self.assertEqual(step_1["status"], "REINFORCED")

        step_2 = engine.step(
            raw_input={"v_1": 3.0, "v_2": 2.0},
            observed_outcome=12.0,
        )

        self.assertEqual(step_2["status"], "EXPLAINED_INTERNALLY")
        self.assertEqual(step_2["loop_executed"], "LOOP_D")
        self.assertEqual(step_2["loops_executed"], ["LOOP_A", "LOOP_B", "LOOP_D"])
        self.assertEqual(step_2["evaluated_module_id"], "mod_only_v1")
        self.assertIsNotNone(step_2["promoted_module_id"])
        self.assertEqual(step_2["active_module_id"], step_2["promoted_module_id"])
        self.assertIsNone(step_2["discovered_dimension"])

        active_modules = engine.memory_repo.get_active_modules()
        self.assertEqual(len(active_modules), 1)
        self.assertEqual(active_modules[0].active_dimensions, {"v_1", "v_2"})

        step_3 = engine.step(
            raw_input={"v_1": 2.0, "v_2": 0.0},
            observed_outcome=4.0,
        )
        self.assertEqual(step_3["status"], "REINFORCED")
        self.assertEqual(step_3["active_module_id"], step_2["promoted_module_id"])

    def test_full_loop_c_and_d_flow_external_discovery_consensus_and_promotion(self) -> None:
        """
        Senaryo 4 (Tam Döngü C ve D Akışı - Dış Keşif, Konsensüs ve H_3 Terfisi):
        İç uzayda açıklanamayan bir anomali oluştuğunda Döngü C yeni boyutu ('theta_kesif')
        keşfetmeli, Ω_global'e eklemeli ve Döngü D yeni modülü H_3'e terfi ettirmelidir.
        """
        seed_module = MicroModule(
            module_id="mod_base_v1",
            active_dimensions={"v_1"},
            confidence=ConfidenceScore(score=0.70),
            predict_fn=lambda vec: vec.values["v_1"] * 2.0,
        )

        providers = [
            SourceProvider(
                source_id="ext_node_1",
                trust_weight=1.0,
                static_payload={"dimension_name": "theta_kesif", "statistic": 2.0},
            ),
            SourceProvider(
                source_id="ext_node_2",
                trust_weight=1.2,
                static_payload={"dimension_name": "theta_kesif", "statistic": 2.05},
            ),
            SourceProvider(
                source_id="ext_node_3",
                trust_weight=0.9,
                static_payload={"dimension_name": "theta_kesif", "statistic": 1.98},
            ),
        ]

        engine = (
            EngineBuilder()
            .with_dimension("v_1", default_value=0.0, variance_threshold=0.05)
            .with_initial_module(seed_module)
            .with_source_provider(providers[0])
            .with_source_provider(providers[1])
            .with_source_provider(providers[2])
            .with_hyperparameters(
                error_threshold=0.05,
                consensus_threshold=0.70,
                regression_tolerance=0.05,
            )
            .build()
        )

        step_1 = engine.step(raw_input={"v_1": 2.0}, observed_outcome=4.0)
        self.assertEqual(step_1["status"], "REINFORCED")
        self.assertFalse(engine.parameter_space.has_dimension("theta_kesif"))

        step_2 = engine.step(raw_input={"v_1": 3.0}, observed_outcome=12.0)

        self.assertEqual(step_2["status"], "PROMOTED_TO_H3")
        self.assertEqual(step_2["loop_executed"], "LOOP_D")
        self.assertEqual(
            step_2["loops_executed"],
            ["LOOP_A", "LOOP_B", "LOOP_C", "LOOP_D"],
        )
        self.assertEqual(step_2["discovered_dimension"], "theta_kesif")
        self.assertIsNotNone(step_2["consensus_score"])
        self.assertGreaterEqual(step_2["consensus_score"], 0.70)
        self.assertIsNotNone(step_2["promoted_module_id"])
        self.assertEqual(step_2["active_module_id"], step_2["promoted_module_id"])

        self.assertTrue(engine.parameter_space.has_dimension("theta_kesif"))

        event_types = [type(e) for e in step_2["events"]]
        self.assertIn(ModuleDemotedEvent, event_types)
        self.assertIn(AnomalyQuarantinedEvent, event_types)
        self.assertIn(DimensionDiscoveredEvent, event_types)
        self.assertIn(ModulePromotedEvent, event_types)

        active_modules = engine.memory_repo.get_active_modules()
        self.assertEqual(len(active_modules), 1)
        self.assertEqual(active_modules[0].module_id, step_2["promoted_module_id"])
        self.assertEqual(active_modules[0].active_dimensions, {"v_1", "theta_kesif"})

    def test_consensus_failure_freezes_hypothesis_in_h2_without_corrupting_h3(self) -> None:
        """
        Senaryo 5 (Konsensüs Yetersizliği Durumu - H_2'de Dondurma):
        Dış kaynaklar çeliştiğinde adım 'FROZEN_IN_H2' statüsü almalı ve H_3 bozulmamalıdır.
        """
        seed_module = MicroModule(
            module_id="mod_strict_v1",
            active_dimensions={"v_1"},
            confidence=ConfidenceScore(score=0.80),
            predict_fn=lambda vec: vec.values["v_1"] * 2.0,
        )

        conflicting_providers = [
            SourceProvider(
                source_id="conflict_1",
                trust_weight=1.0,
                static_payload={"dimension_name": "alpha_dim", "statistic": 0.9},
            ),
            SourceProvider(
                source_id="conflict_2",
                trust_weight=1.0,
                static_payload={"dimension_name": "beta_dim", "statistic": 0.9},
            ),
            SourceProvider(
                source_id="conflict_3",
                trust_weight=1.0,
                static_payload={"dimension_name": "gamma_dim", "statistic": 0.9},
            ),
        ]

        engine = (
            EngineBuilder()
            .with_dimension("v_1", default_value=0.0)
            .with_initial_module(seed_module)
            .with_source_provider(conflicting_providers[0])
            .with_source_provider(conflicting_providers[1])
            .with_source_provider(conflicting_providers[2])
            .with_hyperparameters(
                error_threshold=0.05,
                consensus_threshold=0.75,
            )
            .build()
        )

        telemetry = engine.step(raw_input={"v_1": 4.0}, observed_outcome=50.0)

        self.assertEqual(telemetry["status"], "FROZEN_IN_H2")
        self.assertEqual(telemetry["loop_executed"], "LOOP_C")
        self.assertEqual(telemetry["loops_executed"], ["LOOP_A", "LOOP_B", "LOOP_C"])
        self.assertIsNone(telemetry["active_module_id"])
        self.assertIsNone(telemetry["promoted_module_id"])
        self.assertIsNone(telemetry["discovered_dimension"])
        self.assertIsNotNone(telemetry["consensus_score"])
        self.assertLess(telemetry["consensus_score"], 0.75)

        self.assertEqual(set(engine.parameter_space.dimensions.keys()), {"v_1"})
        self.assertEqual(len(engine.memory_repo.get_active_modules()), 0)
        quarantined_modules = engine.memory_repo.get_quarantined_modules()
        self.assertEqual(len(quarantined_modules), 1)
        self.assertEqual(quarantined_modules[0].module_id, "mod_strict_v1")


if __name__ == "__main__":
    unittest.main()