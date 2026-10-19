"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Sprint 1 Domain İnvaryant Test Süiti.

Bu test modülü; MEMORY.md ve ARCHITECTURE.md anayasasında tanımlanan
epistemik yanılabilirlik (P < 1.0), değişmezlik (immutability),
alt-uzay/boyut bütünlüğü ve mikro-modül yaşam döngüsü kurallarını
sıfır harici kütüphane kullanarak doğrular.

from __**Sprint 1 (Domain Katmanı)** invaryantlarını doğrulayan, sıfır harici kütüphane bağımlılıklı **`tests/unit/test_domain_invariants.py`** birim test süiti aşağıdadır.

python

Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Sprint 1 Domain İnvaryant Test Süiti.

Bu test modülü; MEMORY.md ve ARCHITECTURE.md anayasasında tanımlanan
epistemik yanılabilirlik (P < 1.0), değişmezlik (immutability), alt-uzay
izolasyonu, istisna öncelikli modül değerlendirmesi ve global parametre
uzayı kurallarını yalnızca Python yerleşik 'unittest' kütüphanesiyle doğrular.

Çalıştırma komutu:
    python -m unittest tests/unit/test_domain_invariants.py -v
"""

from __future__ import annotations
from dataclasses import FrozenInstanceError
import math
import unittest

from core.domain import (
    CognitiveDomainException,
    ComponentVector,
    ConfidenceScore,
    DimensionAxis,
    DimensionNotFoundException,
    EpistemicInvariantViolationException,
    GlobalParameterSpace,
    MicroModule,
    ModuleDemotedEvent,
    ModulePromotedEvent,
)


class TestConfidenceScoreInvariants(unittest.TestCase):
    """
    Invariant #1: Epistemik Yanılabilirlik (P < 1.0) ve Asimptotik Güncelleme Testleri.
    """

    def test_rejects_score_greater_than_or_equal_to_one(self) -> None:
        """P >= 1.0 mutlak kesinlik değerleri EpistemicInvariantViolationException fırlatmalıdır."""
        for invalid_score in (1.0, 1.0001, 5.0):
            with self.subTest(score=invalid_score):
                with self.assertRaises(EpistemicInvariantViolationException):
                    ConfidenceScore(score=invalid_score)

    def test_rejects_score_at_or_above_p_max(self) -> None:
        """Yarı-açık [P_min, P_max) aralığı gereği score >= P_max reddedilmelidir."""
        with self.assertRaises(EpistemicInvariantViolationException):
            ConfidenceScore(score=ConfidenceScore.DEFAULT_P_MAX)

        with self.assertRaises(EpistemicInvariantViolationException):
            ConfidenceScore(score=0.99995, p_min=0.0, p_max=0.9999)

    def test_rejects_negative_nan_and_infinite_scores(self) -> None:
        """P < 0.0, NaN ve Inf değerleri EpistemicInvariantViolationException fırlatmalıdır."""
        invalid_values = (-0.0001, -1.0, float("nan"), float("inf"), float("-inf"))
        for val in invalid_values:
            with self.subTest(score=val):
                with self.assertRaises(EpistemicInvariantViolationException):
                    ConfidenceScore(score=val)

    def test_reinforce_follows_asymptotic_formula_and_never_reaches_p_max(self) -> None:
        """
        Asimptotik artış P_{t+1} = P_t + η * (P_max - P_t) formülünü işlemeli ve
        1000 ardışık iterasyon sonunda dahi P_max (0.9999) veya 1.0 değerine ulaşmamalıdır.
        """
        initial_p = 0.5
        eta = 0.25
        p_max = ConfidenceScore.DEFAULT_P_MAX

        confidence = ConfidenceScore(score=initial_p)
        step_one = confidence.reinforce(learning_rate=eta)

        expected_step_one = initial_p + eta * (p_max - initial_p)
        self.assertAlmostEqual(step_one.score, expected_step_one, places=12)

        # 1000 iterasyon boyunca agresif öğrenme oranıyla pekiştirme (IEEE-754 sınır testi)
        current = confidence
        for _ in range(1000):
            next_conf = current.reinforce(learning_rate=0.9)
            self.assertGreaterEqual(next_conf.score, current.score)
            self.assertLess(next_conf.score, p_max)
            self.assertLess(next_conf.score, 1.0)
            current = next_conf

    def test_penalize_preserves_geometric_decay_and_respects_p_min(self) -> None:
        """
        Geometrik ceza P_{t+1} = P_t * γ formülünü korumalı ve P_min altına inmemelidir.
        """
        p_min = 0.05
        confidence = ConfidenceScore(score=0.8, p_min=p_min, p_max=0.9999)

        penalized = confidence.penalize(decay_rate=0.5)
        self.assertAlmostEqual(penalized.score, 0.4, places=12)

        # Çoklu ceza altında p_min tabanının korunması
        current = penalized
        for _ in range(100):
            current = current.penalize(decay_rate=0.1)
            self.assertGreaterEqual(current.score, p_min)

        self.assertEqual(current.score, p_min)

    def test_rejects_invalid_learning_and_decay_rates(self) -> None:
        """η ve γ parametreleri (0.0, 1.0) açık aralığı dışında ise istisna fırlatılmalıdır."""
        confidence = ConfidenceScore(score=0.5)
        for invalid_rate in (-0.1, 0.0, 1.0, 1.5, float("nan")):
            with self.subTest(rate=invalid_rate):
                with self.assertRaises(EpistemicInvariantViolationException):
                    confidence.reinforce(learning_rate=invalid_rate)
                with self.assertRaises(EpistemicInvariantViolationException):
                    confidence.penalize(decay_rate=invalid_rate)


class TestComponentVectorInvariants(unittest.TestCase):
    """
    ComponentVector Değişmezlik (Immutability) ve Alt-Uzay / Delta Operasyon Testleri.
    """

    def test_immutability_and_external_dict_isolation(self) -> None:
        """
        ComponentVector oluşturulduktan sonra öznitelikleri değiştirilememeli ve
        kurucuya verilen dış sözlüğün sonradan değiştirilmesi vektörü etkilememelidir.
        """
        raw_input = {"v_1": 1.0, "v_2": 2.5}
        vector = ComponentVector(values=raw_input)

        # Dış sözlüğü manipüle et
        raw_input["v_1"] = 999.0
        raw_input["v_3"] = -5.0

        self.assertEqual(vector.values["v_1"], 1.0)
        self.assertNotIn("v_3", vector.values)

        # Frozen dataclass atama engeli
        with self.assertRaises(FrozenInstanceError):
            vector.values = {"v_1": 0.0}  # type: ignore[misc]

    def test_extract_subspace_success_and_missing_dimension_exception(self) -> None:
        """
        extract_subspace geçerli alt kümede V_aktif döndürmeli, mevcut olmayan
        boyut talep edildiğinde DimensionNotFoundException fırlatmalıdır.
        """
        vector = ComponentVector(values={"v_1": 10.0, "v_2": 20.0, "v_3": 30.0})
        active_subspace = vector.extract_subspace({"v_1", "v_3"})

        self.assertEqual(active_subspace.dimension_keys, {"v_1", "v_3"})
        self.assertEqual(active_subspace.values["v_1"], 10.0)
        self.assertEqual(active_subspace.values["v_3"], 30.0)

        with self.assertRaises(DimensionNotFoundException):
            vector.extract_subspace({"v_1", "v_non_existent"})

    def test_calculate_delta_success_and_missing_dimension_exception(self) -> None:
        """
        calculate_delta iki vektör arasındaki farkı doğru hesaplamalı, referans
        vektörde eksik boyut varsa DimensionNotFoundException fırlatmalıdır.
        """
        v_prev = ComponentVector(values={"v_1": 4.0, "v_2": 10.0})
        v_curr = ComponentVector(values={"v_1": 6.5, "v_2": 7.0})

        delta = v_curr.calculate_delta(v_prev)
        self.assertAlmostEqual(delta["v_1"], 2.5)
        self.assertAlmostEqual(delta["v_2"], -3.0)

        v_incomplete = ComponentVector(values={"v_1": 4.0})
        with self.assertRaises(DimensionNotFoundException):
            v_curr.calculate_delta(v_incomplete)


class TestMicroModuleInvariants(unittest.TestCase):
    """
    MicroModule Değerlendirme Önceliği, Boyut Kontrolü ve Karantina Tenzil Testleri.
    """

    def test_evaluate_checks_exceptions_before_calling_predict_fn(self) -> None:
        """
        Kayıtlı bir istisna kuralı (exception) eşleştiğinde, ana predict_fn
        hiç çağrılmadan doğrudan istisna sonucu döndürülmelidir.
        """
        predict_called = False

        def base_predictor(vec: ComponentVector) -> float:
            nonlocal predict_called
            predict_called = True
            return vec.values["v_1"] * 2.0

        module = MicroModule(
            module_id="mod_alpha",
            active_dimensions={"v_1"},
            confidence=ConfidenceScore(score=0.75),
            predict_fn=base_predictor,
        )

        # Sınır durumu için istisna kuralı ekle: v_1 < 0 ise sabit -1.0 döndür
        module.attach_exception(
            condition=lambda vec: vec.values["v_1"] < 0.0,
            outcome=-1.0,
        )

        # 1. Durum: İstisna koşulu sağlanıyor -> predict_fn çağrılmamalı
        anomaly_vec = ComponentVector(values={"v_1": -5.0})
        outcome_exc = module.evaluate(anomaly_vec)
        self.assertEqual(outcome_exc, -1.0)
        self.assertFalse(predict_called)

        # 2. Durum: İstisna koşulu sağlanmıyor -> predict_fn çağrılmalı
        normal_vec = ComponentVector(values={"v_1": 4.0})
        outcome_norm = module.evaluate(normal_vec)
        self.assertEqual(outcome_norm, 8.0)
        self.assertTrue(predict_called)

    def test_evaluate_raises_dimension_not_found_when_active_dimension_missing(self) -> None:
        """
        Modülün active_dimensions kümesindeki boyutlardan biri girdi vektöründe
        eksikse DimensionNotFoundException fırlatılmalıdır.
        """
        module = MicroModule(
            module_id="mod_beta",
            active_dimensions={"v_1", "v_2"},
            confidence=ConfidenceScore(score=0.6),
            predict_fn=lambda vec: vec.values["v_1"] + vec.values["v_2"],
        )

        incomplete_vector = ComponentVector(values={"v_1": 1.0})
        with self.assertRaises(DimensionNotFoundException):
            module.evaluate(incomplete_vector)

    def test_demote_to_quarantine_lowers_confidence_and_emits_event(self) -> None:
        """
        demote_to_quarantine çağrıldığında modül güveni γ çarpanı ile düşmeli
        ve H_3 -> H_2 geçişini temsil eden ModuleDemotedEvent üretilmelidir.
        """
        module = MicroModule(
            module_id="mod_gamma",
            active_dimensions={"v_1"},
            confidence=ConfidenceScore(score=0.8),
            predict_fn=lambda vec: vec.values["v_1"],
        )

        event = module.demote_to_quarantine(decay_rate=0.5)

        self.assertIsInstance(event, ModuleDemotedEvent)
        self.assertEqual(event.module_id, "mod_gamma")
        self.assertEqual(event.source_pool, "H_3")
        self.assertEqual(event.target_pool, "H_2")
        self.assertAlmostEqual(module.confidence.score, 0.4, places=12)
        self.assertAlmostEqual(event.confidence.score, 0.4, places=12)

    def test_promote_to_active_emits_promoted_event(self) -> None:
        """promote_to_active metodu H_2 -> H_3 ModulePromotedEvent üretmelidir."""
        module = MicroModule(
            module_id="mod_shadow",
            active_dimensions={"v_1"},
            confidence=ConfidenceScore(score=0.5),
            predict_fn=lambda vec: vec.values["v_1"],
        )

        event = module.promote_to_active()
        self.assertIsInstance(event, ModulePromotedEvent)
        self.assertEqual(event.source_pool, "H_2")
        self.assertEqual(event.target_pool, "H_3")


class TestGlobalParameterSpaceInvariants(unittest.TestCase):
    """
    GlobalParameterSpace (Ω_global) Boyut Tescili ve Pasif Boyut Taraması Testleri.
    """

    def test_register_duplicate_dimension_raises_invariant_violation(self) -> None:
        """
        Ω_global içinde zaten var olan bir boyut ekseni tekrar eklenmek istendiğinde
        EpistemicInvariantViolationException fırlatılmalıdır.
        """
        space = GlobalParameterSpace()
        axis_theta_1 = DimensionAxis(name="theta_1", data_type="continuous", variance_threshold=0.01)

        space.register_dimension(axis_theta_1)
        self.assertTrue(space.has_dimension("theta_1"))

        duplicate_axis = DimensionAxis(name="theta_1", data_type="discrete", variance_threshold=0.05)
        with self.assertRaises(EpistemicInvariantViolationException):
            space.register_dimension(duplicate_axis)

    def test_get_passive_dimensions_returns_exact_set_difference(self) -> None:
        """
        get_passive_dimensions metodu Ω_global \\ V_aktif küme farkını eksiksiz döndürmelidir.
        """
        space = GlobalParameterSpace()
        for dim_name in ("theta_1", "theta_2", "theta_3", "theta_4"):
            space.register_dimension(
                DimensionAxis(name=dim_name, data_type="continuous", variance_threshold=0.001)
            )

        active_vector = ComponentVector(values={"theta_1": 0.5, "theta_3": -1.2})
        passive_dims = space.get_passive_dimensions(active_vector)

        self.assertEqual(passive_dims, {"theta_2", "theta_4"})


if __name__ == "__main__":
    unittest.main()