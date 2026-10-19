"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Katman 4 Dış Araştırma ve Konsensüs Altyapısı.

Bu modül; Katman 3 (Seviye 3: Karşı-Olgusal Teşhis) tarafından üretilen
hedef sorguyu Q(Δ, V_aktif, Y_obs) bağımsız dış kaynaklara yönelten
`MockMultiSourceResearchAgent` adaptörünü ve Döngü C çapraz konsensüs /
veri zehirlenmesi (Byzantine poisoning) korumasını yürüten `CrossConsensusEngine`
sınıfını tanımlar.

Matematiksel Kural (Döngü C - Invariant #5 & #6):
    C(S) = (1 / Σ w_i) * Σ [ w_i * Sim(s_i, s_robust) ]

    - C(S) >= consensus_threshold ve N >= N_min ise yeni boyut (θ_kesif) onaylanır.
    - C(S) < consensus_threshold ise hipotez reddedilmez, H_2'de dondurulur (frozen).

İnşa Kuralları:
- Sıfır Harici Kütüphane (Invariant #4): Yalnızca 'typing', 'math', 'abc', 're', 'json', 'dataclasses'.
- Mutlak Alan Bağımsızlığı (Invariant #3): Domain-agnostic tasarım.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.application.ports import IResearchAgent
from core.domain import (
    CognitiveDomainException,
    EpistemicInvariantViolationException,
    TargetQuery,
)


@dataclass(frozen=True)
class SourceProvider:
    """
    Katman 4 dış araştırmasında kullanılan tekil ve bağımsız bir harici bilgi
    kaynağını (API, Veritabanı, Akademik Korpus vb.) temsil eden değer/sağlayıcı nesnesi.

    Attributes:
        source_id: Kaynağın benzersiz kimliği.
        trust_weight: Kaynağın tarihsel/epistemik güven ağırlığı (w_i > 0.0).
        query_fn: TargetQuery alıp yapılandırılmış aday boyut sözlüğü dönen fonksiyon.
        static_payload: Test veya sabit kaynak senaryoları için JSON/Dict yanıt şablonu.
    """

    source_id: str
    trust_weight: float = 1.0
    query_fn: Optional[Callable[[TargetQuery], Optional[Dict[str, Any]]]] = None
    static_payload: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise CognitiveDomainException(
                "SourceProvider 'source_id' alanı boş olmayan bir metin (str) olmalıdır.",
                context={"source_id": self.source_id},
            )

        if (
            not isinstance(self.trust_weight, (int, float))
            or isinstance(self.trust_weight, bool)
            or math.isnan(float(self.trust_weight))
            or math.isinf(float(self.trust_weight))
            or float(self.trust_weight) <= 0.0
        ):
            raise CognitiveDomainException(
                "SourceProvider 'trust_weight' (w_i) pozitif geçerli bir sayı (> 0.0) olmalıdır.",
                context={"source_id": self.source_id, "trust_weight": self.trust_weight},
            )
        object.__setattr__(self, "trust_weight", float(self.trust_weight))

        if self.query_fn is not None and not callable(self.query_fn):
            raise CognitiveDomainException(
                "SourceProvider 'query_fn' çağrılabilir (Callable) bir fonksiyon olmalıdır.",
                context={"source_id": self.source_id},
            )

        if self.static_payload is not None and not isinstance(self.static_payload, dict):
            raise CognitiveDomainException(
                "SourceProvider 'static_payload' bir sözlük (Dict[str, Any]) olmalıdır.",
                context={"source_id": self.source_id},
            )

        if self.static_payload is not None:
            # Dış mutasyonu önlemek için JSON tabanlı derin kopya
            sanitized_copy = json.loads(json.dumps(self.static_payload))
            object.__setattr__(self, "static_payload", sanitized_copy)

    def fetch_hypothesis(self, query: TargetQuery) -> Optional[Dict[str, Any]]:
        """
        Verilen karşı-olgusal hedef sorguyu (Q) kaynak üzerinde çalıştırır ve
        kaynak kimliği ile ağırlığını içeren aday hipotez sözlüğünü döner.
        """
        if not isinstance(query, TargetQuery):
            raise CognitiveDomainException(
                "SourceProvider.fetch_hypothesis yalnızca bir TargetQuery nesnesi kabul eder.",
                context={"received_type": type(query).__name__},
            )

        raw_response: Optional[Dict[str, Any]] = None
        if self.query_fn is not None:
            raw_response = self.query_fn(query)
        elif self.static_payload is not None:
            raw_response = dict(self.static_payload)

        if raw_response is None:
            return None

        if not isinstance(raw_response, dict):
            raise CognitiveDomainException(
                f"'{self.source_id}' kaynağından dönen yanıt Dict[str, Any] formatında olmalıdır.",
                context={"source_id": self.source_id, "received_type": type(raw_response).__name__},
            )

        enriched = dict(raw_response)
        enriched["source_id"] = self.source_id
        enriched.setdefault("weight", self.trust_weight)
        return enriched


class MockMultiSourceResearchAgent(IResearchAgent):
    """
    Bağımsız harici kaynak sağlayıcılarını (`SourceProvider`) barındıran ve
    `IResearchAgent` SPI portunu gerçekleyen çoklu-kaynak araştırma ajanı.
    """

    def __init__(self, providers: Optional[List[SourceProvider]] = None) -> None:
        self._providers: Dict[str, SourceProvider] = {}
        if providers is not None:
            if not isinstance(providers, list):
                raise CognitiveDomainException(
                    "MockMultiSourceResearchAgent 'providers' parametresi bir liste olmalıdır."
                )
            for provider in providers:
                self.register_provider(provider)

    @property
    def providers(self) -> List[SourceProvider]:
        """Kayıtlı bağımsız kaynak sağlayıcılarının listesini döner."""
        return list(self._providers.values())

    def register_provider(self, provider: SourceProvider) -> None:
        """
        Yeni bir bağımsız veri kaynağını ajana kaydeder.
        Aynı 'source_id' ile mükerrer kayıt yapılmasını (Sybil/Single-Source yanılgısı) engeller.
        """
        if not isinstance(provider, SourceProvider):
            raise CognitiveDomainException(
                "register_provider yalnızca bir SourceProvider nesnesi kabul eder.",
                context={"received_type": type(provider).__name__},
            )

        if provider.source_id in self._providers:
            raise EpistemicInvariantViolationException(
                f"'{provider.source_id}' kimlikli dış kaynak zaten kayıtlı (Bağımsız kaynak ihlali).",
                context={"source_id": provider.source_id},
            )

        self._providers[provider.source_id] = provider

    def remove_provider(self, source_id: str) -> None:
        """Belirtilen kimliğe sahip kaynağı listeden çıkarır."""
        if isinstance(source_id, str):
            self._providers.pop(source_id, None)

    def query_external_sources(self, query: TargetQuery) -> List[Dict[str, Any]]:
        """
        Karşı-olgusal hedef sorguyu Q(Δ, V_aktif, Y_obs) kayıtlı tüm bağımsız
        kaynaklara yöneltir ve yapılandırılmış aday boyut önerilerini toplar.

        Args:
            query: Katman 3 tarafından üretilen TargetQuery nesnesi.

        Returns:
            Bağımsız kaynaklardan dönen hipotez sözlüklerinin listesi.
        """
        if not isinstance(query, TargetQuery):
            raise CognitiveDomainException(
                "query_external_sources 'query' parametresi bir TargetQuery nesnesi olmalıdır.",
                context={"received_type": type(query).__name__},
            )

        collected_responses: List[Dict[str, Any]] = []
        for provider in self._providers.values():
            hypothesis = provider.fetch_hypothesis(query)
            if hypothesis is not None and len(hypothesis) > 0:
                collected_responses.append(hypothesis)

        return collected_responses


class CrossConsensusEngine:
    """
    Katman 4 / Döngü C: Çapraz Konsensüs ve Zehirlenme Koruması (Poisoning Protection) Motoru.

    Bağımsız N kaynaktan toplanan aday boyut hipotezlerini hem semantik (boyut adı /
    kavramsal token uyumu) hem de istatistiksel (Byzantine aykırı değer budamalı
    medyan/MAD yakınlığı) açıdan denetler.

    Matematiksel Formül:
        C(S) = (1 / Σ w_i) * Σ [ w_i * Sim(s_i, s_robust) ]
    """

    _DIMENSION_KEYS: Tuple[str, ...] = (
        "dimension_name",
        "candidate_dimension",
        "dimension",
        "name",
    )
    _METRIC_KEYS: Tuple[str, ...] = (
        "statistic",
        "correlation",
        "confidence",
        "score",
        "effect_size",
        "value",
    )

    def __init__(
        self,
        min_independent_sources: int = 2,
        outlier_scale_floor: float = 0.15,
    ) -> None:
        """
        Args:
            min_independent_sources: Konsensüsün geçerli sayılması için gereken
                minimum bağımsız kaynak sayısı (N_min >= 2, Invariant #5).
            outlier_scale_floor: İstatistiksel benzerlik çekirdeğinde kullanılan
                minimum ölçek tabanı (sıfıra bölünme ve aşırı hassasiyet koruması).
        """
        if (
            not isinstance(min_independent_sources, int)
            or isinstance(min_independent_sources, bool)
            or min_independent_sources < 1
        ):
            raise CognitiveDomainException(
                "CrossConsensusEngine 'min_independent_sources' pozitif bir tam sayı olmalıdır.",
                context={"min_independent_sources": min_independent_sources},
            )

        if (
            not isinstance(outlier_scale_floor, (int, float))
            or isinstance(outlier_scale_floor, bool)
            or math.isnan(float(outlier_scale_floor))
            or math.isinf(float(outlier_scale_floor))
            or float(outlier_scale_floor) <= 0.0
        ):
            raise CognitiveDomainException(
                "CrossConsensusEngine 'outlier_scale_floor' pozitif bir sayı olmalıdır.",
                context={"outlier_scale_floor": outlier_scale_floor},
            )

        self.min_independent_sources: int = min_independent_sources
        self.outlier_scale_floor: float = float(outlier_scale_floor)

    def evaluate_consensus(
        self,
        candidate_responses: List[Dict[str, Any]],
        consensus_threshold: float = 0.70,
    ) -> Tuple[bool, Optional[str], float]:
        """
        Toplanan dış kaynak yanıtları üzerinde zehirlenme korumalı (Byzantine-resilient)
        ağırlıklı çapraz konsensüs denetimi yapar.

        İşleyiş:
        1. Yanıtları doğrular, mükerrer `source_id` kayıtlarını tekilleştirir ve
           minimum bağımsız kaynak sayısını (`min_independent_sources`) denetler.
        2. Ağırlıklı oylama ve token benzerliğiyle baskın aday boyutu (θ_robust) belirler.
        3. Baskın boyutu destekleyen kaynakların sayısal metrikleri üzerinden
           medyan (s_robust) ve Medyan Mutlak Sapma (MAD) hesaplayarak aşırı uç
           (zehirlenmiş / Byzantine) değerleri tespit eder.
        4. Her kaynak için bileşik benzerliği hesaplar:
               Sim(s_i, s_robust) = Sim_sem(d_i, θ_robust) * Sim_stat(v_i, v_robust)
        5. Ağırlıklı konsensüs skoru C(S) >= consensus_threshold ise:
               (True, confirmed_dimension_name, consensus_score)
           aksi halde (H_2'de dondurulmak üzere):
               (False, None, consensus_score)
           döner.

        Args:
            candidate_responses: Dış kaynaklardan dönen yapılandırılmış yanıt listesi.
            consensus_threshold: Onay için gereken minimum konsensüs eşiği ∈ (0.0, 1.0].

        Returns:
            (is_verified, confirmed_dimension_name, consensus_score) üçlü demeti.
        """
        if not isinstance(candidate_responses, list):
            raise CognitiveDomainException(
                "evaluate_consensus 'candidate_responses' parametresi bir liste olmalıdır.",
                context={"received_type": type(candidate_responses).__name__},
            )

        if (
            not isinstance(consensus_threshold, (int, float))
            or isinstance(consensus_threshold, bool)
            or math.isnan(float(consensus_threshold))
            or math.isinf(float(consensus_threshold))
            or not (0.0 < float(consensus_threshold) <= 1.0)
        ):
            raise EpistemicInvariantViolationException(
                "Konsensüs eşiği (consensus_threshold) (0.0, 1.0] aralığında olmalıdır.",
                context={"consensus_threshold": consensus_threshold},
            )

        parsed_records = self._parse_and_deduplicate_responses(candidate_responses)
        if len(parsed_records) < self.min_independent_sources:
            return (False, None, 0.0)

        # 1. Semantik Temsilci Seçimi: Ağırlıklı olarak en çok desteklenen boyut adını bul
        dimension_weights: Dict[str, float] = {}
        canonical_to_original: Dict[str, str] = {}

        for rec in parsed_records:
            canon_name = rec["canonical_name"]
            dimension_weights[canon_name] = dimension_weights.get(canon_name, 0.0) + rec["weight"]
            if canon_name not in canonical_to_original:
                canonical_to_original[canon_name] = rec["raw_name"]

        # En yüksek ağırlığa sahip kanonik boyutu s_robust'un semantik merkezi seç
        robust_canonical_dim = max(
            dimension_weights.keys(),
            key=lambda k: (dimension_weights[k], k),
        )
        confirmed_raw_name = canonical_to_original[robust_canonical_dim]

        # 2. İstatistiksel Dayanıklı Merkez (Byzantine-Resilient Median & MAD)
        supporting_metrics: List[float] = [
            rec["metric"]
            for rec in parsed_records
            if self._compute_semantic_similarity(rec["canonical_name"], robust_canonical_dim) >= 0.5
            and rec["metric"] is not None
        ]

        robust_median: Optional[float] = None
        robust_scale: float = self.outlier_scale_floor
        if supporting_metrics:
            robust_median = self._calculate_median(supporting_metrics)
            abs_deviations = [abs(m - robust_median) for m in supporting_metrics]
            mad = self._calculate_median(abs_deviations)
            # Normal dağılım tutarlılık çarpanı (1.4826 * MAD) ve alt taban koruması
            robust_scale = max(
                1.4826 * mad,
                abs(robust_median) * self.outlier_scale_floor,
                self.outlier_scale_floor,
            )

        # 3. Döngü C Konsensüs Formülü: C(S) = (1 / Σ w_i) * Σ w_i * Sim(s_i, s_robust)
        total_weight: float = 0.0
        weighted_similarity_sum: float = 0.0

        for rec in parsed_records:
            w_i: float = rec["weight"]
            sim_sem: float = self._compute_semantic_similarity(
                rec["canonical_name"], robust_canonical_dim
            )

            sim_stat: float = 1.0
            if robust_median is not None and rec["metric"] is not None:
                sim_stat = self._compute_statistical_similarity(
                    value=rec["metric"],
                    robust_center=robust_median,
                    scale=robust_scale,
                )

            combined_sim: float = sim_sem * sim_stat
            weighted_similarity_sum += w_i * combined_sim
            total_weight += w_i

        if total_weight <= 0.0:
            return (False, None, 0.0)

        consensus_score: float = weighted_similarity_sum / total_weight
        consensus_score = max(0.0, min(1.0, consensus_score))

        if consensus_score >= float(consensus_threshold):
            return (True, confirmed_raw_name, consensus_score)

        return (False, None, consensus_score)

    # =========================================================================
    # DAHİLİ ZEHİRLENME KORUMASI VE İSTATİSTİKSEL YARDIMCILAR
    # =========================================================================

    def _parse_and_deduplicate_responses(
        self,
        responses: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Ham kaynak yanıtlarını doğrular, sayısal ağırlık ve metrikleri çıkarır,
        aynı `source_id` üzerinden gelen mükerrer yanıtları tekilleştirir.
        """
        parsed: List[Dict[str, Any]] = []
        seen_sources: Set[str] = set()

        for idx, item in enumerate(responses):
            if not isinstance(item, dict):
                raise CognitiveDomainException(
                    "Dış kaynak yanıt listesindeki her eleman bir sözlük (Dict[str, Any]) olmalıdır.",
                    context={"index": idx, "received_type": type(item).__name__},
                )

            # Boyut adını çıkar
            raw_dim_name: Optional[str] = None
            for key in self._DIMENSION_KEYS:
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    raw_dim_name = val.strip()
                    break

            if raw_dim_name is None:
                continue

            # Kaynak kimliğini belirle ve mükerrer kaynakları filtrele
            source_id = str(item.get("source_id", f"anon_source_{idx}")).strip()
            if not source_id:
                source_id = f"anon_source_{idx}"
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)

            # Güven ağırlığını (w_i) doğrula
            raw_weight = item.get("weight", item.get("trust_weight", 1.0))
            if (
                not isinstance(raw_weight, (int, float))
                or isinstance(raw_weight, bool)
                or math.isnan(float(raw_weight))
                or math.isinf(float(raw_weight))
                or float(raw_weight) <= 0.0
            ):
                raise CognitiveDomainException(
                    f"'{source_id}' kaynağının ağırlığı (weight) pozitif geçerli bir sayı olmalıdır.",
                    context={"source_id": source_id, "weight": raw_weight},
                )

            # İstatistiksel metriği (varsa) çıkar ve doğrula
            metric_val: Optional[float] = None
            for m_key in self._METRIC_KEYS:
                if m_key in item and item[m_key] is not None:
                    raw_metric = item[m_key]
                    if (
                        isinstance(raw_metric, (int, float))
                        and not isinstance(raw_metric, bool)
                    ):
                        f_metric = float(raw_metric)
                        if math.isnan(f_metric) or math.isinf(f_metric):
                            raise CognitiveDomainException(
                                f"'{source_id}' kaynağının '{m_key}' metriği NaN veya Inf olamaz.",
                                context={"source_id": source_id, "metric": raw_metric},
                            )
                        metric_val = f_metric
                        break

            parsed.append(
                {
                    "source_id": source_id,
                    "raw_name": raw_dim_name,
                    "canonical_name": self._canonicalize_name(raw_dim_name),
                    "weight": float(raw_weight),
                    "metric": metric_val,
                }
            )

        return parsed

    @staticmethod
    def _canonicalize_name(name: str) -> str:
        """
        Boyut adını küçük harfe çevirir ve alfanümerik olmayan ayraçları
        standart alt çizgiye ('_') dönüştürür.
        """
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
        return cleaned if cleaned else name.strip().lower()

    @classmethod
    def _compute_semantic_similarity(cls, name_a: str, name_b: str) -> float:
        """
        İki kanonik boyut adı arasındaki semantik/kavramsal benzerliği [0.0, 1.0]
        aralığında hesaplar. Tam eşleşmede 1.0, alt-token örtüşmesinde Jaccard
        benzerliği döner.
        """
        if name_a == name_b:
            return 1.0

        tokens_a = {t for t in name_a.split("_") if t}
        tokens_b = {t for t in name_b.split("_") if t}
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return float(intersection) / float(union) if union > 0 else 0.0

    @staticmethod
    def _compute_statistical_similarity(
        value: float,
        robust_center: float,
        scale: float,
    ) -> float:
        """
        Kaynağın bildirdiği istatistiksel değer (v_i) ile dayanıklı medyan merkezi
        (s_robust) arasındaki uyumu hesaplar.

        Byzantine Zehirlenme Koruması:
        Merkezden 3 ölçekten (3 * scale) daha uzak olan aşırı uç (poisoned/outlier)
        değerler 0.0 benzerlik alarak konsensüs dışına itilir; yakın değerler ise
        Gaussian çekirdeği exp(-0.5 * (z / 2)^2) ile [0.0, 1.0] aralığında puanlanır.
        """
        distance = abs(value - robust_center)
        z_score = distance / scale

        # Aşırı uç Byzantine zehirlenme kesmesi (Hard Byzantine Cutoff)
        if z_score > 3.0:
            return 0.0

        return math.exp(-0.5 * ((z_score / 2.0) ** 2))

    @staticmethod
    def _calculate_median(values: List[float]) -> float:
        """Verilen sayısal listenin medyan (ortanca) değerini hesaplar."""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        if n % 2 == 1:
            return sorted_vals[mid]
        return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


__all__ = [
    "SourceProvider",
    "MockMultiSourceResearchAgent",
    "CrossConsensusEngine",
]