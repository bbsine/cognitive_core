"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Sprint 4 Dış Araştırma ve Konsensüs Test Süiti.

Bu test modülü; Katman 4 (Döngü C: Çapraz Konsensüs & Zehirlenme Koruması)
kapsamındaki SourceProvider, MockMultiSourceResearchAgent ve CrossConsensusEngine
sınıflarını; Sybil saldırısı engelleme, semantik/istatistiksel uzlaşı, Byzantine
aykırı değer budama (poisoning mitigation) ve minimum bağımsız kaynak (N >= N_min)
invaryantları açısından yalnızca Python standart kütüphanesiyle doğrular.

Çalıştırma komutu:
    python -m unittest tests/unit/test_research_and_consensus.py -v
"""

from __future__ import annotations
from typing import Any, Dict, List
import unittest

from core.domain import (
    CognitiveDomainException,
    ComponentVector,
    EpistemicInvariantViolationException,
    TargetQuery,
)
from core.infrastructure.research_agent import (
    CrossConsensusEngine,
    MockMultiSourceResearchAgent,
    SourceProvider,
)


class TestMockMultiSourceResearchAgent(unittest.TestCase):
    """
    SourceProvider ve MockMultiSourceResearchAgent (Katman 4 Dış Sorgu Yönlendirme) Testleri.
    """

    def setUp(self) -> None:
        self.sample_query = TargetQuery(
            delta_state={"v_1": 2.5},
            active_vector=ComponentVector(values={"v_1": 5.0}),
            observed_outcome=42.0,
        )

    def test_single_provider_fetch_hypothesis_with_static_and_dynamic_fn(self) -> None:
        """
        SourceProvider hem statik payload hem de dinamik query_fn üzerinden
        kaynak kimliği (source_id) ve güven ağırlığı (weight) eklenmiş hipotez dönmelidir.
        """
        static_provider = SourceProvider(
            source_id="src_static_1",
            trust_weight=1.25,
            static_payload={"dimension_name": "theta_kesif", "correlation": 0.88},
        )
        static_res = static_provider.fetch_hypothesis(self.sample_query)
        self.assertIsNotNone(static_res)
        assert static_res is not None
        self.assertEqual(static_res["source_id"], "src_static_1")
        self.assertAlmostEqual(static_res["weight"], 1.25)
        self.assertEqual(static_res["dimension_name"], "theta_kesif")

        dynamic_provider = SourceProvider(
            source_id="src_dynamic_1",
            trust_weight=0.90,
            query_fn=lambda q: {
                "dimension_name": "theta_dyn",
                "statistic": q.delta_state["v_1"] * 0.2,
            },
        )
        dyn_res = dynamic_provider.fetch_hypothesis(self.sample_query)
        self.assertIsNotNone(dyn_res)
        assert dyn_res is not None
        self.assertEqual(dyn_res["source_id"], "src_dynamic_1")
        self.assertAlmostEqual(dyn_res["weight"], 0.90)
        self.assertAlmostEqual(dyn_res["statistic"], 0.50)

    def test_duplicate_source_id_raises_epistemic_invariant_violation(self) -> None:
        """
        Aynı source_id ile birden fazla kaynak kaydedilmek istendiğinde (Sybil / Tekil Kaynak
        yanılgısı) EpistemicInvariantViolationException fırlatılmalıdır (Invariant #5).
        """
        agent = MockMultiSourceResearchAgent()
        provider_a = SourceProvider(
            source_id="independent_node_1",
            trust_weight=1.0,
            static_payload={"dimension_name": "theta_1"},
        )
        provider_duplicate = SourceProvider(
            source_id="independent_node_1",
            trust_weight=2.0,
            static_payload={"dimension_name": "theta_2"},
        )

        agent.register_provider(provider_a)
        with self.assertRaises(EpistemicInvariantViolationException):
            agent.register_provider(provider_duplicate)

    def test_query_external_sources_aggregates_valid_hypotheses(self) -> None:
        """
        query_external_sources çağrısı kayıtlı tüm aktif kaynakları sorgulamalı,
        None dönen kaynakları filtreleyerek yapılandırılmış yanıt listesini dönmelidir.
        """
        providers = [
            SourceProvider(
                source_id="src_1",
                trust_weight=1.0,
                static_payload={"dimension_name": "theta_kesif", "correlation": 0.85},
            ),
            SourceProvider(
                source_id="src_2",
                trust_weight=1.5,
                query_fn=lambda q: {"dimension_name": "theta_kesif", "correlation": 0.82},
            ),
            SourceProvider(
                source_id="src_silent",
                trust_weight=1.0,
                query_fn=lambda q: None,
            ),
        ]

        agent = MockMultiSourceResearchAgent(providers=providers)
        responses = agent.query_external_sources(self.sample_query)

        self.assertEqual(len(responses), 2)
        self.assertEqual({r["source_id"] for r in responses}, {"src_1", "src_2"})


class TestCrossConsensusEngine(unittest.TestCase):
    """
    CrossConsensusEngine (Döngü C: Konsensüs Doğrulama ve Byzantine Zehirlenme Koruması) Testleri.
    """

    def setUp(self) -> None:
        self.engine = CrossConsensusEngine(
            min_independent_sources=2,
            outlier_scale_floor=0.15,
        )

    def test_consensus_success_confirms_discovered_dimension(self) -> None:
        """
        Senaryo 2 (Konsensüs Başarısı):
        Bağımsız kaynaklar aynı/benzer boyutu ('theta_kesif') tutarlı istatistiksel
        metriklerle bildirdiğinde C(S) >= consensus_threshold sağlanmalı ve
        (True, 'theta_kesif', score) dönmelidir.
        """
        candidate_responses: List[Dict[str, Any]] = [
            {
                "source_id": "src_alpha",
                "dimension_name": "theta_kesif",
                "correlation": 0.84,
                "weight": 1.0,
            },
            {
                "source_id": "src_beta",
                "dimension_name": "theta_kesif",
                "correlation": 0.86,
                "weight": 1.2,
            },
            {
                "source_id": "src_gamma",
                "dimension_name": "theta_kesif",
                "correlation": 0.85,
                "weight": 0.8,
            },
        ]

        is_verified, confirmed_dim, score = self.engine.evaluate_consensus(
            candidate_responses=candidate_responses,
            consensus_threshold=0.75,
        )

        self.assertTrue(is_verified)
        self.assertEqual(confirmed_dim, "theta_kesif")
        self.assertGreaterEqual(score, 0.75)
        self.assertLessEqual(score, 1.0)

    def test_consensus_failure_freezes_hypothesis_when_sources_conflict(self) -> None:
        """
        Senaryo 3 (Yetersiz Konsensüs / H_2'de Dondurma):
        Kaynaklar arasında semantik çelişki olduğunda konsensüs skoru eşik altında
        kalmalı ve (False, None, score) dönmelidir.
        """
        conflicting_responses: List[Dict[str, Any]] = [
            {
                "source_id": "src_1",
                "dimension_name": "theta_alpha",
                "correlation": 0.80,
                "weight": 1.0,
            },
            {
                "source_id": "src_2",
                "dimension_name": "omega_beta",
                "correlation": 0.80,
                "weight": 1.0,
            },
            {
                "source_id": "src_3",
                "dimension_name": "sigma_gamma",
                "correlation": 0.80,
                "weight": 1.0,
            },
        ]

        is_verified, confirmed_dim, score = self.engine.evaluate_consensus(
            candidate_responses=conflicting_responses,
            consensus_threshold=0.70,
        )

        self.assertFalse(is_verified)
        self.assertIsNone(confirmed_dim)
        self.assertLess(score, 0.70)
        self.assertAlmostEqual(score, 1.0 / 3.0, places=4)

    def test_byzantine_poisoning_attack_prunes_extreme_outlier_and_preserves_consensus(self) -> None:
        """
        Senaryo 4 (Byzantine / Zehirlenme Koruması):
        Çoğunluk (4 dürüst kaynak) ~0.80 metriği üzerinde uzlaşırken, 1 kötü niyetli
        (Byzantine) kaynak 1000.0 gibi aşırı uç (outlier) bir değer bildirdiğinde:
        - Dayanıklı medyan (s_robust) bozulmamalı,
        - Zehirli kaynağın benzerliği 0.0'a budanmalı,
        - Kalan 4 dürüst kaynağın ağırlıklı konsensüsü (~0.80) eşiği (0.70) geçerek
          doğru boyutu ('theta_kesif') onaylamalıdır.
        """
        responses_with_byzantine_node: List[Dict[str, Any]] = [
            {
                "source_id": "honest_1",
                "dimension_name": "theta_kesif",
                "statistic": 0.79,
                "weight": 1.0,
            },
            {
                "source_id": "honest_2",
                "dimension_name": "theta_kesif",
                "statistic": 0.80,
                "weight": 1.0,
            },
            {
                "source_id": "honest_3",
                "dimension_name": "theta_kesif",
                "statistic": 0.81,
                "weight": 1.0,
            },
            {
                "source_id": "honest_4",
                "dimension_name": "theta_kesif",
                "statistic": 0.80,
                "weight": 1.0,
            },
            {
                "source_id": "byzantine_attacker",
                "dimension_name": "theta_kesif",
                "statistic": 1000.0,  # Aşırı uç veri zehirleme saldırısı
                "weight": 1.0,
            },
        ]

        is_verified, confirmed_dim, score = self.engine.evaluate_consensus(
            candidate_responses=responses_with_byzantine_node,
            consensus_threshold=0.70,
        )

        # 4 dürüst kaynak ~1.0 benzerlik alır, 1 Byzantine kaynak 0.0 alır -> Skor ~ 4/5 = 0.80
        self.assertTrue(is_verified)
        self.assertEqual(confirmed_dim, "theta_kesif")
        self.assertGreaterEqual(score, 0.75)
        self.assertLessEqual(score, 0.81)

    def test_minimum_independent_sources_requirement_and_sybil_deduplication(self) -> None:
        """
        Senaryo 5 (Minimum Bağımsız Kaynak Şartı - N < N_min):
        - Tek bir kaynak (N=1 < N_min=2) ne kadar yüksek ağırlığa sahip olursa olsun
          konsensüs doğrudan (False, None, 0.0) dönmelidir (Invariant #5).
        - Aynı source_id ile gönderilen mükerrer yanıtlar tekilleştirilmeli ve
          N < N_min nedeniyle reddedilmelidir.
        """
        single_source_response = [
            {
                "source_id": "sole_oracle",
                "dimension_name": "theta_kesif",
                "correlation": 0.99,
                "weight": 100.0,
            }
        ]
        is_verified, confirmed_dim, score = self.engine.evaluate_consensus(
            candidate_responses=single_source_response,
            consensus_threshold=0.50,
        )
        self.assertFalse(is_verified)
        self.assertIsNone(confirmed_dim)
        self.assertEqual(score, 0.0)

        # Sybil denemesi: Aynı source_id'ye sahip 3 yanıt
        sybil_responses = [
            {"source_id": "same_node", "dimension_name": "theta_kesif", "correlation": 0.90},
            {"source_id": "same_node", "dimension_name": "theta_kesif", "correlation": 0.91},
            {"source_id": "same_node", "dimension_name": "theta_kesif", "correlation": 0.89},
        ]
        sybil_verified, sybil_dim, sybil_score = self.engine.evaluate_consensus(
            candidate_responses=sybil_responses,
            consensus_threshold=0.50,
        )
        self.assertFalse(sybil_verified)
        self.assertIsNone(sybil_dim)
        self.assertEqual(sybil_score, 0.0)

    def test_invalid_threshold_and_corrupted_metrics_raise_domain_exceptions(self) -> None:
        """
        Geçersiz konsensüs eşiğinde EpistemicInvariantViolationException,
        NaN/Inf metriklerde ise CognitiveDomainException fırlatılmalıdır.
        """
        valid_pair = [
            {"source_id": "s1", "dimension_name": "theta_1", "statistic": 0.5},
            {"source_id": "s2", "dimension_name": "theta_1", "statistic": 0.5},
        ]

        for bad_threshold in (0.0, -0.5, 1.05, float("nan")):
            with self.subTest(threshold=bad_threshold):
                with self.assertRaises(EpistemicInvariantViolationException):
                    self.engine.evaluate_consensus(valid_pair, consensus_threshold=bad_threshold)

        corrupted_responses = [
            {"source_id": "s1", "dimension_name": "theta_1", "statistic": float("nan")},
            {"source_id": "s2", "dimension_name": "theta_1", "statistic": 0.5},
        ]
        with self.assertRaises(CognitiveDomainException):
            self.engine.evaluate_consensus(corrupted_responses, consensus_threshold=0.70)


if __name__ == "__main__":
    unittest.main()