"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Sinir Ağı ve Hesaplama Motoru Altyapı Adaptörü.

Bu modül; Application katmanındaki INeuralEnginePort SPI sözleşmesini hiçbir harici
kütüphane (NumPy, PyTorch vb.) kullanmadan, tamamen saf Python, analitik Ridge/OLS
çözücüsü ve Normalized Least Mean Squares (NLMS) optimizasyonu ile gerçekleyen
hesaplama adaptörünü tanımlar.
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Set, Tuple
import uuid

from core.application.ports import INeuralEnginePort
from core.domain import (
    CognitiveDomainException,
    ComponentVector,
    ConfidenceScore,
    MicroModule,
    Observation,
    TargetQuery,
)


class PurePythonNeuralEngine(INeuralEnginePort):
    """
    INeuralEnginePort arayüzünün sıfır harici bağımlılıklı, deterministik
    saf Python (Pure Python) implementasyonu.
    """

    def __init__(
        self,
        initial_confidence: float = 0.50,
        training_epochs: int = 250,
        step_size: float = 0.50,
        regularization: float = 1e-6,
        use_mse: bool = True,
        default_activation_radius: float = 10.0,
    ) -> None:
        self._initial_confidence_vo = ConfidenceScore(score=initial_confidence)

        if not isinstance(training_epochs, int) or isinstance(training_epochs, bool) or training_epochs <= 0:
            raise CognitiveDomainException(
                "PurePythonNeuralEngine 'training_epochs' pozitif bir tam sayı olmalıdır.",
                context={"training_epochs": training_epochs},
            )

        if (
            not isinstance(step_size, (int, float))
            or isinstance(step_size, bool)
            or math.isnan(float(step_size))
            or math.isinf(float(step_size))
            or not (0.0 < float(step_size) <= 2.0)
        ):
            raise CognitiveDomainException(
                "PurePythonNeuralEngine 'step_size' (0.0, 2.0] aralığında geçerli bir sayı olmalıdır.",
                context={"step_size": step_size},
            )

        if (
            not isinstance(regularization, (int, float))
            or isinstance(regularization, bool)
            or math.isnan(float(regularization))
            or math.isinf(float(regularization))
            or float(regularization) < 0.0
        ):
            raise CognitiveDomainException(
                "PurePythonNeuralEngine 'regularization' negatif olmayan bir sayı olmalıdır.",
                context={"regularization": regularization},
            )

        self.initial_confidence: float = self._initial_confidence_vo.score
        self.training_epochs: int = training_epochs
        self.step_size: float = float(step_size)
        self.regularization: float = float(regularization)
        self.use_mse: bool = bool(use_mse)
        self.default_activation_radius: float = max(0.5, float(default_activation_radius))

    def decompose_input(self, raw_input: Dict[str, Any]) -> ComponentVector:
        """
        Katman 1 - Nitelik Ayrıştırma Operatörü Φ(X):
        Ham girdi sözlüğündeki yalnızca geçerli sayısal (int, float) değerleri
        filtreler, bool/metin/None değerleri yoksayar ve değişmez bir
        ComponentVector (V) nesnesine dönüştürür.
        """
        if not isinstance(raw_input, dict):
            raise CognitiveDomainException(
                "decompose_input 'raw_input' parametresi bir sözlük (Dict[str, Any]) olmalıdır.",
                context={"received_type": type(raw_input).__name__},
            )

        extracted_features: Dict[str, float] = {}
        for key, value in raw_input.items():
            if not isinstance(key, str) or not key.strip():
                continue

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue

            float_val = float(value)
            if math.isnan(float_val) or math.isinf(float_val):
                raise CognitiveDomainException(
                    f"Ham girdideki '{key}' boyutu geçersiz sayısal değer (NaN/Inf) içeriyor.",
                    context={"key": key, "value": value},
                )

            extracted_features[key] = float_val

        if not extracted_features:
            raise CognitiveDomainException(
                "Nitelik ayrıştırma (Φ) sonucunda ham girdide hiçbir geçerli sayısal boyut bulunamadı.",
                context={"raw_keys": list(raw_input.keys())},
            )

        return ComponentVector(values=extracted_features)

    def synthesize_candidate_module(
        self,
        target_query: TargetQuery,
        context_data: List[Observation],
    ) -> MicroModule:
        """
        Katman 5 - Neurogenesis ve İç Simülasyon Ön-Eğitimi:
        Hedef sorgu Q(Δ, V_aktif, Y_obs) ve bağlamsal gözlem verilerini (context_data)
        kullanarak genişletilmiş boyut uzayında Dim(V_aktif ∪ {θ_kesif}) çalışan,
        `centroid_vector` ve `activation_radius` değerleri hesaplanmış deterministik
        bir aday `MicroModule` (m_yeni) sentezler.
        """
        if not isinstance(target_query, TargetQuery):
            raise CognitiveDomainException(
                "synthesize_candidate_module 'target_query' parametresi bir TargetQuery olmalıdır.",
                context={"received_type": type(target_query).__name__},
            )

        if not isinstance(context_data, list):
            raise CognitiveDomainException(
                "synthesize_candidate_module 'context_data' parametresi bir liste (List[Observation]) olmalıdır.",
                context={"received_type": type(context_data).__name__},
            )

        active_dimensions: Set[str] = set(target_query.active_vector.dimension_keys) | set(
            target_query.delta_state.keys()
        )

        valid_observations: List[Observation] = []
        for obs in context_data:
            if not isinstance(obs, Observation):
                raise CognitiveDomainException(
                    "context_data içindeki tüm elemanlar Observation nesnesi olmalıdır.",
                    context={"invalid_element": obs},
                )
            valid_observations.append(obs)
            active_dimensions.update(obs.features.dimension_keys)

        sorted_dims: List[str] = sorted(active_dimensions)

        training_samples: List[Tuple[Dict[str, float], float]] = []
        for obs in valid_observations:
            if (
                obs.observed_outcome is not None
                and isinstance(obs.observed_outcome, (int, float))
                and not isinstance(obs.observed_outcome, bool)
            ):
                y_val = float(obs.observed_outcome)
                if not (math.isnan(y_val) or math.isinf(y_val)):
                    sample_vec = {
                        dim: obs.features.values.get(dim, 0.0) for dim in sorted_dims
                    }
                    training_samples.append((sample_vec, y_val))

        if (
            isinstance(target_query.observed_outcome, (int, float))
            and not isinstance(target_query.observed_outcome, bool)
        ):
            target_y = float(target_query.observed_outcome)
            if not (math.isnan(target_y) or math.isinf(target_y)):
                query_vec = {
                    dim: target_query.active_vector.values.get(
                        dim, target_query.delta_state.get(dim, 0.0)
                    )
                    for dim in sorted_dims
                }
                training_samples.append((query_vec, target_y))

        weights, bias = self._optimize_linear_parameters(sorted_dims, training_samples)
        centroid_vec, radius = self._compute_regime_centroid_and_radius(
            sorted_dims, training_samples
        )

        frozen_weights: Dict[str, float] = dict(weights)
        frozen_bias: float = float(bias)
        fallback_outcome: Any = target_query.observed_outcome

        def synthesized_predictor(vector: ComponentVector) -> Any:
            if not training_samples:
                return fallback_outcome
            prediction_val = frozen_bias
            for dim_name, weight in frozen_weights.items():
                prediction_val += weight * vector.values.get(dim_name, 0.0)
            return prediction_val

        setattr(synthesized_predictor, "weights", dict(frozen_weights))
        setattr(synthesized_predictor, "bias", frozen_bias)

        return MicroModule(
            module_id=f"m_syn_{uuid.uuid4().hex[:12]}",
            active_dimensions=active_dimensions,
            confidence=ConfidenceScore(score=self.initial_confidence),
            predict_fn=synthesized_predictor,
            created_at=time.time(),
            centroid_vector=centroid_vec,
            activation_radius=radius,
        )

    def evaluate_regression(
        self,
        candidate: MicroModule,
        anchor_data: List[Observation],
        tolerance: float,
    ) -> bool:
        """
        Katman 5 / Döngü D - Regresyon ve Stabilite Testi (Geriye Dönük Koruma):
        H_2 gölge dalındaki aday modülün (m_yeni), kendi aktivasyon bölgesiyle
        uyumlu çapa verileri (anchor_data) üzerindeki ortalama kaybını hesaplar.
        """
        if not isinstance(candidate, MicroModule):
            raise CognitiveDomainException(
                "evaluate_regression 'candidate' parametresi bir MicroModule nesnesi olmalıdır.",
                context={"received_type": type(candidate).__name__},
            )

        if not isinstance(anchor_data, list):
            raise CognitiveDomainException(
                "evaluate_regression 'anchor_data' parametresi bir Observation listesi olmalıdır.",
                context={"received_type": type(anchor_data).__name__},
            )

        if (
            not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or math.isnan(float(tolerance))
            or math.isinf(float(tolerance))
            or float(tolerance) < 0.0
        ):
            raise CognitiveDomainException(
                "evaluate_regression 'tolerance' negatif olmayan geçerli bir sayı olmalıdır.",
                context={"tolerance": tolerance},
            )

        if not anchor_data:
            return True

        # Aday modülün bir uzmanlık merkezi (centroid_vector) varsa, yalnızca
        # modülün kendi aktivasyon yarıçapı (activation_radius) içine düşen
        # çapa verilerini filtrele; hiçbir çapa düşmüyorsa tüm çapa verilerini değerlendir.
        relevant_anchors: List[Observation] = []
        for obs in anchor_data:
            if not isinstance(obs, Observation):
                raise CognitiveDomainException(
                    "anchor_data içindeki tüm elemanlar Observation nesnesi olmalıdır.",
                    context={"invalid_element": obs},
                )
            if obs.observed_outcome is None:
                continue

            if candidate.centroid_vector is not None and not math.isinf(candidate.activation_radius):
                dist = self._euclidean_distance(
                    obs.features,
                    candidate.centroid_vector,
                    candidate.active_dimensions,
                )
                if dist <= candidate.activation_radius:
                    relevant_anchors.append(obs)
            else:
                relevant_anchors.append(obs)

        eval_pool = relevant_anchors if relevant_anchors else [
            o for o in anchor_data if o.observed_outcome is not None
        ]

        if not eval_pool:
            return True

        total_loss: float = 0.0
        evaluated_count: int = 0

        for obs in eval_pool:
            projected_values: Dict[str, float] = {
                dim: obs.features.values.get(dim, 0.0)
                for dim in candidate.active_dimensions
            }
            projected_vector = ComponentVector(values=projected_values)

            try:
                predicted_outcome = candidate.evaluate(projected_vector)
            except CognitiveDomainException:
                return False

            sample_error = self._compute_sample_loss(
                observed=obs.observed_outcome,
                predicted=predicted_outcome,
            )
            if math.isnan(sample_error) or math.isinf(sample_error):
                return False

            total_loss += sample_error
            evaluated_count += 1

        if evaluated_count == 0:
            return True

        mean_loss: float = total_loss / evaluated_count
        return mean_loss <= float(tolerance)

    # =========================================================================
    # DAHİLİ SAYISAL OPTİMİZASYON VE REJİM MERKEZİ YARDIMCILARI (PURE PYTHON)
    # =========================================================================

    def _optimize_linear_parameters(
        self,
        dimensions: List[str],
        samples: List[Tuple[Dict[str, float], float]],
    ) -> Tuple[Dict[str, float], float]:
        """
        Önce analitik L2-düzenlileştirilmiş En Küçük Kareler (Ridge / OLS Gauss
        Eliminasyonu) ile tam çözümü hesaplar; tekil veya yetersiz durumda NLMS
        iterasyonuna başvurur.
        """
        weights: Dict[str, float] = {dim: 0.0 for dim in dimensions}
        bias: float = 0.0

        if not samples or not dimensions:
            return weights, bias

        # 1. Önce sıfır sapmalı (homogeneous: b = 0) analitik Ridge çözümünü dene.
        hom_weights = self._solve_ridge_system(
            dimensions=dimensions,
            samples=samples,
            include_bias=False,
        )
        if hom_weights is not None:
            hom_w, _ = hom_weights
            hom_mse = self._evaluate_fit_mse(dimensions, samples, hom_w, 0.0)
            if hom_mse <= 1e-7:
                return hom_w, 0.0

        # 2. Sapmalı (affine: y = w^T x + b) analitik Ridge çözümünü hesapla.
        affine_solution = self._solve_ridge_system(
            dimensions=dimensions,
            samples=samples,
            include_bias=True,
        )
        if affine_solution is not None:
            aff_w, aff_b = affine_solution
            aff_mse = self._evaluate_fit_mse(dimensions, samples, aff_w, aff_b)
            if hom_weights is not None:
                hom_w, _ = hom_weights
                hom_mse = self._evaluate_fit_mse(dimensions, samples, hom_w, 0.0)
                if hom_mse <= aff_mse:
                    return hom_w, 0.0
            if aff_mse <= 1e-4:
                return aff_w, aff_b

        # 3. Yedek: NLMS İteratif Optimizasyon
        for _ in range(self.training_epochs):
            max_abs_error: float = 0.0
            for x_dict, y_true in samples:
                y_hat: float = bias
                norm_sq: float = 1.0 + self.regularization

                for dim in dimensions:
                    x_val = x_dict.get(dim, 0.0)
                    y_hat += weights[dim] * x_val
                    norm_sq += x_val * x_val

                error: float = y_true - y_hat
                abs_err = abs(error)
                if abs_err > max_abs_error:
                    max_abs_error = abs_err

                step_factor: float = (self.step_size * error) / norm_sq
                decay_factor: float = 1.0 - (self.regularization * 0.01)

                for dim in dimensions:
                    x_val = x_dict.get(dim, 0.0)
                    weights[dim] = (weights[dim] * decay_factor) + (step_factor * x_val)

                bias += step_factor

            if max_abs_error <= 1e-12:
                break

        return weights, bias

    def _solve_ridge_system(
        self,
        dimensions: List[str],
        samples: List[Tuple[Dict[str, float], float]],
        include_bias: bool,
    ) -> Optional[Tuple[Dict[str, float], float]]:
        """
        Saf Python Gauss Eliminasyonu (kısmi pivotlamalı) kullanarak
        (X^T X + λI) θ = X^T y doğrusal sistemini çözer.
        """
        d = len(dimensions)
        p = d + (1 if include_bias else 0)
        reg = max(1e-9, self.regularization * 1e-3)

        a_mat: List[List[float]] = [[0.0] * p for _ in range(p)]
        b_vec: List[float] = [0.0] * p

        for x_dict, y_val in samples:
            row = [x_dict.get(dim, 0.0) for dim in dimensions]
            if include_bias:
                row.append(1.0)
            for i in range(p):
                b_vec[i] += row[i] * y_val
                for j in range(p):
                    a_mat[i][j] += row[i] * row[j]

        for i in range(p):
            a_mat[i][i] += reg

        # Gauss Eliminasyonu (Partial Pivoting)
        for col in range(p):
            pivot_row = max(range(col, p), key=lambda r: abs(a_mat[r][col]))
            if abs(a_mat[pivot_row][col]) <= 1e-15:
                return None
            if pivot_row != col:
                a_mat[col], a_mat[pivot_row] = a_mat[pivot_row], a_mat[col]
                b_vec[col], b_vec[pivot_row] = b_vec[pivot_row], b_vec[col]

            pivot_val = a_mat[col][col]
            for r in range(col + 1, p):
                factor = a_mat[r][col] / pivot_val
                b_vec[r] -= factor * b_vec[col]
                for c in range(col, p):
                    a_mat[r][c] -= factor * a_mat[col][c]

        theta: List[float] = [0.0] * p
        for r in range(p - 1, -1, -1):
            s = b_vec[r]
            for c in range(r + 1, p):
                s -= a_mat[r][c] * theta[c]
            theta[r] = s / a_mat[r][r]

        solved_weights = {dimensions[i]: theta[i] for i in range(d)}
        solved_bias = theta[d] if include_bias else 0.0
        return solved_weights, solved_bias

    @staticmethod
    def _evaluate_fit_mse(
        dimensions: List[str],
        samples: List[Tuple[Dict[str, float], float]],
        weights: Dict[str, float],
        bias: float,
    ) -> float:
        if not samples:
            return 0.0
        total_sq = 0.0
        for x_dict, y_true in samples:
            pred = bias + sum(weights[d] * x_dict.get(d, 0.0) for d in dimensions)
            diff = y_true - pred
            total_sq += diff * diff
        return total_sq / len(samples)

    def _compute_regime_centroid_and_radius(
        self,
        dimensions: List[str],
        samples: List[Tuple[Dict[str, float], float]],
    ) -> Tuple[ComponentVector, float]:
        """
        Eğitim örneklerinin koordinat ortalamasından rejim merkezini (`centroid_vector`)
        ve örneklerin merkezden yayılımından `activation_radius` değerini hesaplar.
        """
        if not samples:
            return ComponentVector(values={d: 0.0 for d in dimensions}), self.default_activation_radius

        n = float(len(samples))
        mean_dict: Dict[str, float] = {d: 0.0 for d in dimensions}
        for x_dict, _ in samples:
            for d in dimensions:
                mean_dict[d] += x_dict.get(d, 0.0) / n

        centroid = ComponentVector(values=mean_dict)
        max_dist = 0.0
        for x_dict, _ in samples:
            dist_sq = sum((x_dict.get(d, 0.0) - mean_dict[d]) ** 2 for d in dimensions)
            dist = math.sqrt(dist_sq)
            if dist > max_dist:
                max_dist = dist

        radius = max(self.default_activation_radius, max_dist * 2.5)
        return centroid, radius

    @staticmethod
    def _euclidean_distance(
        vec_a: ComponentVector,
        vec_b: ComponentVector,
        dimensions: Set[str],
    ) -> float:
        return math.sqrt(
            sum(
                (vec_a.values.get(k, 0.0) - vec_b.values.get(k, 0.0)) ** 2
                for k in dimensions
            )
        )

    def _compute_sample_loss(self, observed: Any, predicted: Any) -> float:
        if (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and isinstance(predicted, (int, float))
            and not isinstance(predicted, bool)
        ):
            diff = abs(float(observed) - float(predicted))
            return (diff * diff) if self.use_mse else diff

        return 0.0 if observed == predicted else 1.0


NeuralEngineAdapter = PurePythonNeuralEngine
LinearNeuralEngineAdapter = PurePythonNeuralEngine

__all__ = [
    "PurePythonNeuralEngine",
    "NeuralEngineAdapter",
    "LinearNeuralEngineAdapter",
]