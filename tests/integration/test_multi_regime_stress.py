"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - 1000 Adımlık Çoklu Rejim Stres Testi.

Doğrulanan Mimari Kriterler:
1. Eksik Boyut İmpütasyonu: Kayıtlı bir boyut girdide eksik olduğunda `default_value`
   değerine otomatik hizalanır (`GlobalParameterSpace.align_vector`).
2. Gürültü ve Histerezis Kalkanı (`anomaly_patience=3`): Tekil gürültü sapmalarında
   LOOP_C / LOOP_D tetiklenmez, mevcut modül korunur (`HYSTERESIS_SHIELDED`).
3. 1000 Adımlık Faz 1 -> Faz 2 -> Faz 3 -> Faz 1 Stres Testi:
   - LOOP_D (nörogenez) oranı %5'in altında kalır (yalnızca rejim kırılmalarında çalışır).
   - Faz 3'ten tekrar Faz 1'e dönüldüğünde `ModuleRouter` eski Faz 1 uzman modülünü
     seçerek sıfıra yakın hatayla tahmin üretir (Catastrophic Forgetting = 0).

Çalıştırma komutu:
    python -m unittest tests/integration/test_multi_regime_stress.py -v
"""

from __future__ import annotations
import unittest

from core.domain import (
    ComponentVector,
    ConfidenceScore,
    DimensionAxis,
    GlobalParameterSpace,
    MicroModule,
)
from core.infrastructure import ModuleRouter
from core.infrastructure.research_agent import SourceProvider
from core.interfaces import EngineBuilder


class TestMultiRegimeStressAndRouting(unittest.TestCase):
    """
    Eksik Boyut İmpütasyonu, ModuleRouter ve 1000 Adımlık Çoklu Rejim Stres Testleri.
    """

    def test_missing_dimension_imputation_via_align_vector(self) -> None:
        """
        Gelen girdide kayıtlı bir boyut eksikse, GlobalParameterSpace.align_vector
        hata fırlatmak yerine eksenin default_value değerini atamalıdır.
        """
        space = GlobalParameterSpace()
        space.register_dimension(
            DimensionAxis(name="dim_alpha", data_type="continuous", default_value=0.0)
        )
        space.register_dimension(
            DimensionAxis(name="dim_beta", data_type="continuous", default_value=-2.5)
        )
        space.register_dimension(
            DimensionAxis(name="dim_gamma", data_type="continuous", default_value=1.0)
        )

        aligned = space.align_vector({"dim_alpha": 4.0})
        self.assertEqual(aligned.dimension_keys, {"dim_alpha", "dim_beta", "dim_gamma"})
        self.assertAlmostEqual(aligned.values["dim_alpha"], 4.0)
        self.assertAlmostEqual(aligned.values["dim_beta"], -2.5)
        self.assertAlmostEqual(aligned.values["dim_gamma"], 1.0)

    def test_module_router_selects_closest_specialist_by_euclidean_and_cosine(self) -> None:
        """
        ModuleRouter, H_3 modülleri arasından girdi vektörüne en yakın uzmanlık
        merkezine (centroid_vector) sahip modülü seçmelidir.
        """
        mod_regime_1 = MicroModule(
            module_id="mod_regime_1",
            active_dimensions={"dim_alpha", "dim_beta"},
            confidence=ConfidenceScore(score=0.85),
            predict_fn=lambda v: v.values["dim_alpha"] * 2.0,
            centroid_vector=ComponentVector(values={"dim_alpha": 2.0, "dim_beta": 0.0}),
            activation_radius=4.0,
        )
        mod_regime_2 = MicroModule(
            module_id="mod_regime_2",
            active_dimensions={"dim_alpha", "dim_beta"},
            confidence=ConfidenceScore(score=0.85),
            predict_fn=lambda v: v.values["dim_alpha"] * 2.0 + v.values["dim_beta"] * 5.0,
            centroid_vector=ComponentVector(values={"dim_alpha": 2.0, "dim_beta": 10.0}),
            activation_radius=4.0,
        )

        router = ModuleRouter(similarity_metric="euclidean", min_confidence_threshold=0.50)
        pool = [mod_regime_1, mod_regime_2]

        vec_near_1 = ComponentVector(values={"dim_alpha": 2.1, "dim_beta": 0.2})
        routed_1 = router.route(vec_near_1, candidates=pool)
        self.assertIsNotNone(routed_1)
        assert routed_1 is not None
        self.assertEqual(routed_1.module_id, "mod_regime_1")

        vec_near_2 = ComponentVector(values={"dim_alpha": 1.9, "dim_beta": 9.8})
        routed_2 = router.route(vec_near_2, candidates=pool)
        self.assertIsNotNone(routed_2)
        assert routed_2 is not None
        self.assertEqual(routed_2.module_id, "mod_regime_2")

    def test_1000_step_multi_regime_stress_test_zero_catastrophic_forgetting(self) -> None:
        """
        1000 Adımlık Faz Geçişli ve Gürültülü Stres Testi:
        - Adım 0..249   (Faz 1 - Düz Rejim): y = 2.0 * dim_alpha (dim_beta ve dim_gamma eksik/0.0)
        - Adım 250..499 (Faz 2 - İkinci Rejim): y = 2.0 * dim_alpha + 4.0 * dim_beta
        """