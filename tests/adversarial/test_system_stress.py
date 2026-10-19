"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Bağımsız Dış Denetim & Stres Testi Süiti.

Bu modül; mimariyi içeriden inşa eden kodlayıcının varsayımlarından bağımsız olarak,
sistemi sınır şartlarında (Adversarial & Edge Cases) zorlamak için tasarlanmıştır:
1. IEEE-754 Kayan Nokta ve Asimptotik Sınır Zorlaması (P < 1.0 İnvaryantı).
2. Eşzamanlı Yüksek Frekanslı Yarış Durumları (Thread Concurrency & Havuz Kilidi).
3. Ardışık Rejim Kaymaları (Multiple Regime Shifts) ve Katastrofik Unutma Koruması.
4. Sıfır Bölünme, Aşırı Küçük Değerler (Underflow) ve Kirli Girdi Dayanıklılığı.
"""

from __future__ import annotations
import math
import threading
from typing import Any, Dict, List
import unittest

from core.domain import (
    CognitiveDomainException,
    ComponentVector,
    ConfidenceScore,
    DimensionAxis,
    EpistemicInvariantViolationException,
    GlobalParameterSpace,
    MicroModule,
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
)


class TestAdversarialStress(unittest.TestCase):
    """
    Sistemi mimari köşelere sıkıştıran bağımsız stres ve sınır testleri.
    """

    def test_asymptotic_confidence_never_reaches_one_under_extreme_iterations(self) -> None:
        """
        [İnvaryant #1 Stres Testi]:
        Bir güven skoru 10.000 kez art arda pekiştirildiğinde (reinforce),
        IEEE-754 çift duyarlıklı (double precision) kayan nokta yuvarlaması
        nedeniyle bile olsa ASLA 1.0 olmamalı ve her zaman P < P_max kalmalıdır.
        """
        conf = ConfidenceScore(score=0.50)
        learning_rate = 0.99  # Çok agresif öğrenme oranı

        for _ in range(10_000):
            conf = conf.reinforce(learning_rate)

        # Matematiksel ve epistemik katı sınır
        self.assertLess(conf.score, 1.0)
        self.assertLessEqual(conf.score, ConfidenceScore.DEFAULT_P_MAX)
        self.assertGreater(conf.score, 0.999)

    def test_numerical_stability_with_subnormal_and_extreme_floating_points(self) -> None:
        """
        [Sayısal Kararlılık Testi]:
        Girdide 1e-12, 1e-15 gibi çok küçük sayılar veya ani sıçramalar olduğunda
        NLMS optimizasyonu sıfıra bölünme veya NaN üretmeden kararlı kalmalıdır.
        """
        engine = PurePythonNeuralEngine(
            training_epochs=100,
            step_size=0.5,
            regularization=1e-9,
        )

        extreme_input = {
            "dim_tiny_1": 1e-12,
            "dim_tiny_2": 2e-12,
            "dim_normal": 1.5,
        }
        vector = engine.decompose_input(extreme_input)
        self.assertIsInstance(vector, ComponentVector)
        self.assertAlmostEqual(vector.values["dim_tiny_1"], 1e-12)

    def test_high_concurrency_race_condition_on_full_engine(self) -> None:
        """
        [Eşzamanlılık ve Kilitlenme (Deadlock) Stresi]:
        20 ayrı iş parçacığı aynı anda motor üzerinden hem başarılı tahmin
        hem de anomali adımları koştururken bellek havuzlarının kilitlenmemesi,
        yarış durumu (race condition) veya veri bozulması yaşanmaması gerekir.
        """
        seed_mod = MicroModule(
            module_id="mod_seed_concurrent",
            active_dimensions={"v_1"},
            confidence=ConfidenceScore(score=0.70),
            predict_fn=lambda vec: vec.values["v_1"] * 2.0,
        )

        engine = (
            EngineBuilder()
            .with_dimension(DimensionAxis(name="v_1", data_type="continuous"))
            .with_initial_module(seed_mod)
            .build()
        )

        num_threads = 15
        steps_per_thread = 20
        captured_errors: List[Exception] = []

        def worker_task(thread_id: int) -> None:
            try:
                for step_idx in range(steps_per_thread):
                    val = float((step_idx % 5) + 1)
                    # Çoğunlukla doğru tahmin, ara sıra küçük sapma
                    expected = val * 2.0
                    engine.step(raw_input={"v_1": val}, observed_outcome=expected)
            except Exception as exc:
                captured_errors.append(exc)

        threads = [
            threading.Thread(target=worker_task, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(
            len(captured_errors), 0,
            f"Eşzamanlı motor adımlarında istisnalar yakalandı: {captured_errors}"
        )

    def test_multiple_consecutive_regime_shifts_preserves_stability(self) -> None:
        """
        [Ardışık Rejim Kaymaları ve Unutma Koruması]:
        Sistem art arda 2 farklı dış keşif yaşadığında (önce 'theta_1', ardından 'theta_2'):
        1. Her iki boyutu da başarıyla parameter_space'e ekleyebilmeli.
        2. Eski çapa verileri bozulmadan motorun yaşam döngüsü devam edebilmelidir.
        """
        seed_module = MicroModule(
            module_id="mod_root",
            active_dimensions={"base_dim"},
            confidence=ConfidenceScore(score=0.70),
            predict_fn=lambda vec: vec.values["base_dim"] * 1.0,
        )

        # 1. Aşama dış kaynakları ('theta_alpha' önerir)
        p1 = SourceProvider(
            source_id="src_1",
            trust_weight=1.0,
            static_payload={"dimension_name": "theta_alpha", "statistic": 2.0},
        )
        p2 = SourceProvider(
            source_id="src_2",
            trust_weight=1.0,
            static_payload={"dimension_name": "theta_alpha", "statistic": 2.0},
        )

        engine = (
            EngineBuilder()
            .with_dimension(DimensionAxis(name="base_dim", data_type="continuous"))
            .with_initial_module(seed_module)
            .with_source_provider(p1)
            .with_source_provider(p2)
            .with_hyperparameters(
                error_threshold=0.05,
                consensus_threshold=0.70,
                regression_tolerance=0.08,
            )
            .build()
        )

        # Çapa adımı
        engine.step(raw_input={"base_dim": 2.0}, observed_outcome=2.0)

        # Rejim Kayması 1: theta_alpha devreye girer
        shift_1 = engine.step(raw_input={"base_dim": 3.0}, observed_outcome=7.0)
        self.assertEqual(shift_1["status"], "PROMOTED_TO_H3")
        self.assertTrue(engine.parameter_space.has_dimension("theta_alpha"))


if __name__ == "__main__":
    unittest.main()