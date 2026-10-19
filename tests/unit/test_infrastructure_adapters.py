"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Sprint 3 Altyapı Adaptörleri Test Süiti.

Bu test modülü; 'core.infrastructure' paketindeki InMemoryMemoryRepository ve
PurePythonNeuralEngine SPI adaptörlerinin işlevselliğini, eşzamanlılık (thread-safety)
güvenliğini, JSON durum kalıcılığını ve matematiksel yakınsama/regresyon kurallarını
yalnızca Python standart kütüphanesi kullanarak doğrular.

Çalıştırma komutu:
    python -m unittest tests/unit/test_infrastructure_adapters.py -v
"""

from __future__ import annotations
from pathlib import Path
import tempfile
import threading
from typing import List
import unittest

from core.domain import (
    CognitiveDomainException,
    ComponentVector,
    ConfidenceScore,
    MicroModule,
    Observation,
    TargetQuery,
)
from core.infrastructure import (
    InMemoryMemoryRepository,
    PurePythonNeuralEngine,
)


class TestInMemoryMemoryRepository(unittest.TestCase):
    """
    InMemoryMemoryRepository (H_1, H_2, H_3 Havuzları, JSON Kalıcılık ve Thread-Safety) Testleri.
    """

    def setUp(self) -> None:
        self.repo = InMemoryMemoryRepository()

    def test_h1_observation_ordering_and_limit(self) -> None:
        """
        H_1 gözlem havuzu eklenme sırasını korumalı ve get_recent_observations(limit)
        çağrıldığında en güncelden (son eklenenden) eskiye doğru sıralı döndürmelidir.
        """
        for idx in range(1, 6):
            obs = Observation(
                id=f"obs_{idx}",
                features=ComponentVector(values={"v_1": float(idx)}),
                observed_outcome=float(idx * 10),
                timestamp=1000.0 + idx,
            )
            self.repo.save_observation(obs)

        recent_three = self.repo.get_recent_observations(limit=3)
        self.assertEqual(len(recent_three), 3)
        self.assertEqual([o.id for o in recent_three], ["obs_5", "obs_4", "obs_3"])

        # Limit <= 0 durumunda boş liste dönmeli
        self.assertEqual(self.repo.get_recent_observations(limit=0), [])
        self.assertEqual(self.repo.get_recent_observations(limit=-5), [])

    def test_h2_and_h3_module_isolation_and_safe_removal(self) -> None:
        """
        H_2 (Karantina) ve H_3 (Aktif) havuzları birbirinden izole çalışmalı;
        mevcut olmayan bir modül silinmek istendiğinde hata fırlatmadan güvenle geçmelidir.
        """
        mod_h2 = MicroModule(
            module_id="mod_quarantined_1",
            active_dimensions={"v_1"},
            confidence=ConfidenceScore(score=0.35),
            predict_fn=lambda vec: vec.values["v_1"],
        )
        mod_h3 = MicroModule(
            module_id="mod_active_1",
            active_dimensions={"v_1", "v_2"},
            confidence=ConfidenceScore(score=0.85),
            predict_fn=lambda vec: vec.values["v_1"] + vec.values["v_2"],
        )

        self.repo.save_quarantined_module(mod_h2)
        self.repo.save_active_module(mod_h3)

        # Havuz izolasyonu kontrolü
        quarantined_list = self.repo.get_quarantined_modules()
        active_list = self.repo.get_active_modules()

        self.assertEqual(len(quarantined_list), 1)
        self.assertEqual(quarantined_list[0].module_id, "mod_quarantined_1")
        self.assertEqual(len(active_list), 1)
        self.assertEqual(active_list[0].module_id, "mod_active_1")

        # Mevcut olmayan modül silme denemesi (Sessiz ve güvenli geçiş)
        self.repo.remove_quarantined_module("non_existent_id")
        self.repo.remove_active_module("non_existent_id")
        self.repo.remove_quarantined_module("")

        self.assertEqual(len(self.repo.get_quarantined_modules()), 1)
        self.assertEqual(len(self.repo.get_active_modules()), 1)

        # Gerçek silme işlemi
        self.repo.remove_quarantined_module("mod_quarantined_1")
        self.repo.remove_active_module("mod_active_1")

        self.assertEqual(len(self.repo.get_quarantined_modules()), 0)
        self.assertEqual(len(self.repo.get_active_modules()), 0)

    def test_json_snapshot_dump_clear_and_load_roundtrip(self) -> None:
        """
        dump_to_file ile diske yazılan H_1, H_2 ve H_3 durumu, clear_all ile RAM
        sıfırlandıktan sonra load_from_file ile eksiksiz geri yüklenebilmelidir.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_file = Path(tmp_dir) / "epistemic_state.json"

            obs_1 = Observation(
                id="obs_alpha",
                features=ComponentVector(values={"v_1": 2.5, "v_2": -1.5}),
                observed_outcome=12.0,
                timestamp=1700000001.0,
            )
            mod_q = MicroModule(
                module_id="mod_q_alpha",
                active_dimensions={"v_1"},
                confidence=ConfidenceScore(score=0.25),
                predict_fn=lambda vec: vec.values["v_1"] * 3.0,
            )
            mod_a = MicroModule(
                module_id="mod_a_beta",
                active_dimensions={"v_1", "v_2"},
                confidence=ConfidenceScore(score=0.90),
                predict_fn=lambda vec: vec.values["v_1"] * 4.0 + vec.values["v_2"],
            )

            self.repo.save_observation(obs_1)
            self.repo.save_quarantined_module(mod_q)
            self.repo.save_active_module(mod_a)

            # 1. Durumu diske kaydet
            written_path = self.repo.dump_to_file(snapshot_file)
            self.assertTrue(written_path.exists())

            # 2. RAM'deki tüm havuzları temizle ve boşaldığını doğrula
            self.repo.clear_all()
            self.assertEqual(len(self.repo.get_recent_observations(limit=10)), 0)
            self.assertEqual(len(self.repo.get_quarantined_modules()), 0)
            self.assertEqual(len(self.repo.get_active_modules()), 0)

            # 3. Özel tahmin fonksiyonu eşlemesiyle diskten geri yükle
            self.repo.load_from_file(
                file_path=snapshot_file,
                predictor_registry={
                    "mod_a_beta": lambda vec: vec.values["v_1"] * 4.0 + vec.values["v_2"]
                },
            )

            # 4. H_1, H_2 ve H_3 verilerinin eksiksiz geri geldiğini doğrula
            restored_obs = self.repo.get_recent_observations(limit=10)
            self.assertEqual(len(restored_obs), 1)
            self.assertEqual(restored_obs[0].id, "obs_alpha")
            self.assertAlmostEqual(restored_obs[0].features.values["v_1"], 2.5)
            self.assertEqual(restored_obs[0].observed_outcome, 12.0)

            restored_q = self.repo.get_quarantined_modules()
            self.assertEqual(len(restored_q), 1)
            self.assertEqual(restored_q[0].module_id, "mod_q_alpha")
            self.assertAlmostEqual(restored_q[0].confidence.score, 0.25)

            restored_a = self.repo.get_active_modules()
            self.assertEqual(len(restored_a), 1)
            self.assertEqual(restored_a[0].module_id, "mod_a_beta")
            self.assertEqual(restored_a[0].active_dimensions, {"v_1", "v_2"})
            self.assertAlmostEqual(restored_a[0].confidence.score, 0.90)

            # Geri yüklenen modülün tahmin fonksiyonunun çalıştığını doğrula
            test_vec = ComponentVector(values={"v_1": 3.0, "v_2": 2.0})
            self.assertAlmostEqual(restored_a[0].evaluate(test_vec), 14.0)

    def test_thread_safety_under_concurrent_writes(self) -> None:
        """
        Çoklu iş parçacığı (threading) aynı anda H_1, H_2 ve H_3 havuzlarına
        veri yazdığında yarış durumu (race condition) oluşmamalı ve veri bütünlüğü korunmalıdır.
        """
        num_threads = 10
        items_per_thread = 25
        errors: List[Exception] = []

        def worker(thread_idx: int) -> None:
            try:
                for item_idx in range(items_per_thread):
                    uid = f"t{thread_idx}_i{item_idx}"
                    obs = Observation(
                        id=f"obs_{uid}",
                        features=ComponentVector(values={"v_1": float(item_idx)}),
                        observed_outcome=float(item_idx),
                    )
                    self.repo.save_observation(obs)

                    mod_q = MicroModule(
                        module_id=f"q_{uid}",
                        active_dimensions={"v_1"},
                        confidence=ConfidenceScore(score=0.4),
                        predict_fn=lambda vec: vec.values["v_1"],
                    )
                    self.repo.save_quarantined_module(mod_q)

                    mod_a = MicroModule(
                        module_id=f"a_{uid}",
                        active_dimensions={"v_1"},
                        confidence=ConfidenceScore(score=0.7),
                        predict_fn=lambda vec: vec.values["v_1"],
                    )
                    self.repo.save_active_module(mod_a)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Eşzamanlı yazma sırasında hatalar oluştu: {errors}")

        expected_total = num_threads * items_per_thread
        all_obs = self.repo.get_recent_observations(limit=expected_total + 100)
        self.assertEqual(len(all_obs), expected_total)
        self.assertEqual(len(self.repo.get_quarantined_modules()), expected_total)
        self.assertEqual(len(self.repo.get_active_modules()), expected_total)


class TestPurePythonNeuralEngine(unittest.TestCase):
    """
    PurePythonNeuralEngine (Φ Ayrıştırma, Neurogenesis Ön-Eğitim ve Döngü D Regresyon) Testleri.
    """

    def setUp(self) -> None:
        self.engine = PurePythonNeuralEngine(
            initial_confidence=0.55,
            training_epochs=300,
            step_size=0.50,
            regularization=1e-6,
            use_mse=True,
        )

    def test_decompose_input_filters_booleans_and_non_numerics(self) -> None:
        """
        decompose_input (Φ) metodu bool, str, None ve karmaşık nesneleri dışlamalı,
        yalnızca geçerli int ve float değerleri ComponentVector'a dönüştürmelidir.
        """
        raw_input = {
            "v_1": 10,
            "v_2": -3.75,
            "flag_true": True,
            "flag_false": False,
            "label": "ignored_text",
            "empty": None,
            "nested": {"a": 1},
        }

        vector = self.engine.decompose_input(raw_input)
        self.assertIsInstance(vector, ComponentVector)
        self.assertEqual(vector.dimension_keys, {"v_1", "v_2"})
        self.assertAlmostEqual(vector.values["v_1"], 10.0)
        self.assertAlmostEqual(vector.values["v_2"], -3.75)

    def test_decompose_input_raises_on_nan_inf_or_empty_numeric_set(self) -> None:
        """
        Ham girdide NaN/Inf bulunduğunda veya hiçbir geçerli sayısal özellik
        kalmadığında CognitiveDomainException fırlatılmalıdır.
        """
        with self.assertRaises(CognitiveDomainException):
            self.engine.decompose_input({"v_1": float("nan"), "v_2": 1.0})

        with self.assertRaises(CognitiveDomainException):
            self.engine.decompose_input({"v_1": float("inf")})

        with self.assertRaises(CognitiveDomainException):
            self.engine.decompose_input({"only_bool": True, "only_str": "test"})

    def test_synthesize_candidate_module_neurogenesis_and_convergence(self) -> None:
        """
        synthesize_candidate_module çağrısı, hedef sorgu (Q) ve bağlam gözlemleri
        üzerinden doğrusal ilişkiyi (örn. y = 2*v_1 + 3*theta_new) öğrenen ve
        tahmin üretebilen yeni bir MicroModule (m_yeni) sentezlemelidir.
        """
        # Gizli kural: y = 2.0 * v_1 + 3.0 * theta_new
        context_observations: List[Observation] = [
            Observation(
                id="ctx_1",
                features=ComponentVector(values={"v_1": 1.0, "theta_new": 1.0}),
                observed_outcome=5.0,
            ),
            Observation(
                id="ctx_2",
                features=ComponentVector(values={"v_1": 2.0, "theta_new": 1.0}),
                observed_outcome=7.0,
            ),
            Observation(
                id="ctx_3",
                features=ComponentVector(values={"v_1": 1.0, "theta_new": 3.0}),
                observed_outcome=11.0,
            ),
            Observation(
                id="ctx_4",
                features=ComponentVector(values={"v_1": 3.0, "theta_new": 2.0}),
                observed_outcome=12.0,
            ),
        ]

        target_query = TargetQuery(
            delta_state={"v_1": 1.0, "theta_new": 4.0},
            active_vector=ComponentVector(values={"v_1": 2.0, "theta_new": 4.0}),
            observed_outcome=16.0,  # 2.0*2.0 + 3.0*4.0 = 16.0
        )

        candidate = self.engine.synthesize_candidate_module(
            target_query=target_query,
            context_data=context_observations,
        )

        self.assertIsInstance(candidate, MicroModule)
        self.assertEqual(candidate.active_dimensions, {"v_1", "theta_new"})
        self.assertAlmostEqual(candidate.confidence.score, 0.55)
        self.assertLess(candidate.confidence.score, 1.0)

        # Eğitimde görülmeyen yeni bir test vektöründe tahmin doğruluğunu kontrol et:
        # v_1 = 4.0, theta_new = 2.0 -> Beklenen y = 2.0*4.0 + 3.0*2.0 = 14.0
        unseen_vector = ComponentVector(values={"v_1": 4.0, "theta_new": 2.0})
        prediction = candidate.evaluate(unseen_vector)
        self.assertAlmostEqual(prediction, 14.0, delta=0.15)

    def test_evaluate_regression_passes_on_stable_and_fails_on_forgetting(self) -> None:
        """
        Döngü D Regresyon Testi:
        - Geçmiş çapa verilerini koruyan (Loss <= tolerance) aday modül için True dönmelidir.
        - Geçmiş çapa verilerini bozan (Loss > tolerance) aday modül için False dönmelidir.
        """
        anchor_data: List[Observation] = [
            Observation(
                id="anc_1",
                features=ComponentVector(values={"v_1": 2.0}),
                observed_outcome=4.0,
            ),
            Observation(
                id="anc_2",
                features=ComponentVector(values={"v_1": 5.0}),
                observed_outcome=10.0,
            ),
            Observation(
                id="anc_3",
                features=ComponentVector(values={"v_1": 8.0}),
                observed_outcome=16.0,
            ),
        ]

        # 1. Çapa verileriyle uyumlu modül (y = 2 * v_1 + 0.0 * theta_new)
        compatible_candidate = MicroModule(
            module_id="mod_compatible",
            active_dimensions={"v_1", "theta_new"},
            confidence=ConfidenceScore(score=0.5),
            predict_fn=lambda vec: (vec.values["v_1"] * 2.0) + (vec.values["theta_new"] * 5.0),
        )

        passed = self.engine.evaluate_regression(
            candidate=compatible_candidate,
            anchor_data=anchor_data,
            tolerance=0.01,
        )
        self.assertTrue(passed)

        # 2. Çapa verilerini bozan (Catastrophic Forgetting) modül
        corrupting_candidate = MicroModule(
            module_id="mod_corrupting",
            active_dimensions={"v_1"},
            confidence=ConfidenceScore(score=0.5),
            predict_fn=lambda vec: vec.values["v_1"] * -3.0,
        )

        failed = self.engine.evaluate_regression(
            candidate=corrupting_candidate,
            anchor_data=anchor_data,
            tolerance=0.05,
        )
        self.assertFalse(failed)


if __name__ == "__main__":
    unittest.main()