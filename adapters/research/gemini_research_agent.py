"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Dış Araştırma Ajanı Adaptörü (RESEARCH_SPEC.md).

Bu modül; çekirdek (`core/`) koduna dokunmadan `core.application.ports.IResearchAgent`
SPI sözleşmesini uygulayan, dış dünyaya (Gemini API, harici LLM uç noktası veya
enjekte edilebilir yerel/mock istemci) bağlanarak Katman 4 / Döngü C için yeni
soyut boyut hipotezlerini çeken `GeminiResearchAgent` adaptörünü tanımlar.

Temel Kurallar (RESEARCH_SPEC.md & Invariant #3):
1. Dokunulmaz Çekirdek: `core/` altındaki hiçbir sınıf veya metot değiştirilmez.
2. Mutlak Alan Bağımsızlığı: İstem (prompt) ve şema tamamen soyut vektör/boyut
   kavramlarıyla çalışır:
   {
       "dimension_name": "latent_feature_k",  # str, snake_case
       "statistic": 1.25,                     # float (tahmini ölçek / büyüklük)
       "confidence": 0.95,                    # float, [0.0, 1.0) aralığında (1.0 -> 0.999 kırpılır)
       "source_id": "gemini_api"              # str
   }
3. Hata Dayanıklılığı: Ağ hatası, zaman aşımı, API hatası veya bozuk JSON durumunda
   sistem asla çökmez; `investigate()` güvenli şekilde `None` döner.
"""

from __future__ import annotations
import json
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional
import urllib.error
import urllib.request

from core.application.ports import IResearchAgent
from core.domain import TargetQuery


class GeminiResearchAgent(IResearchAgent):
    """
    `IResearchAgent` SPI sözleşmesini uygulayan, Gemini REST API veya enjekte
    edilen özel bir LLM/Mock istemcisi üzerinden karşı-olgusal anomali sorgularını
    (`TargetQuery`) alan-bağımsız olarak araştıran dış adaptör.
    """

    DEFAULT_SOURCE_ID: str = "gemini_api"
    DEFAULT_MODEL: str = "gemini-flash-latest"
    MAX_ALLOWED_CONFIDENCE: float = 0.999

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        source_id: str = DEFAULT_SOURCE_ID,
        timeout_seconds: float = 10.0,
        transport_fn: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        peer_agents: Optional[List[Callable[[TargetQuery], Optional[Dict[str, Any]]]]] = None,
    ) -> None:
        """
        Args:
            api_key: Gemini API anahtarı. Verilmezse `GEMINI_API_KEY` ortam değişkenine bakılır.
            model_name: Kullanılacak model kimliği (Varsayılan: `gemini-2.5-flash`).
            source_id: Dönen şemada yer alacak kaynak kimliği (Varsayılan: `gemini_api`).
            timeout_seconds: HTTP ağ isteği zaman aşımı süresi (saniye).
            transport_fn: Test, mock veya yerel model entegrasyonu için dışarıdan
                enjekte edilebilen `(prompt, query_payload) -> raw_response_str_or_dict`
                çağrılabilir fonksiyonu.
            peer_agents: Çoklu kaynak / Döngü C konsensüsü için `query_external_sources`
                çağrısında birlikte koşturulacak ek bağımsız ajan fonksiyonları.
        """
        self.api_key: Optional[str] = (
            api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        )
        self.model_name: str = str(model_name).strip() or self.DEFAULT_MODEL
        self.source_id: str = str(source_id).strip() or self.DEFAULT_SOURCE_ID
        self.timeout_seconds: float = max(0.1, float(timeout_seconds))
        self._transport_fn: Optional[Callable[[str, Dict[str, Any]], Any]] = transport_fn
        self._peer_agents: List[Callable[[TargetQuery], Optional[Dict[str, Any]]]] = (
            list(peer_agents) if peer_agents is not None else []
        )

    def investigate(self, query: TargetQuery) -> Optional[Dict[str, Any]]:
        """
        Çekirdek Seviye 3 (Karşı-Olgusal Teşhis) aşamasından gelen `TargetQuery`
        nesnesini inceler, dış modeli sorgular ve doğrulanmış standart şemayı döner.

        Herhangi bir ağ hatası, API başarısızlığı, bozuk JSON veya eksik alan
        durumunda sistemi çökertmez; `None` döner.

        Args:
            query: Anomali farkını (Δ), aktif alt-uzay vektörünü (V_aktif) ve aday boyut
                bilgisini taşıyan `TargetQuery` nesnesi.

        Returns:
            Standart hipotez sözlüğü veya hata durumunda `None`.
        """
        if query is None:
            return None

        try:
            extracted_payload = self._extract_query_fields(query)
            prompt = self._build_investigation_prompt(extracted_payload)

            raw_output = self._dispatch_request(prompt, extracted_payload)
            if raw_output is None:
                return None

            parsed_dict = self._parse_llm_output(raw_output)
            if parsed_dict is None:
                return None

            return self._validate_and_normalize_schema(parsed_dict)
        except Exception:
            # Sınır Kuralı: Dış kaynak hataları çekirdeği asla çökertmemelidir.
            return None

    def query_external_sources(self, query: TargetQuery) -> List[Dict[str, Any]]:
        """
        `core.application.ports.IResearchAgent` SPI sözleşmesinin zorunlu metodu.

        Birincil `investigate(query)` çağrısını ve varsa kayıtlı ek bağımsız kaynakları
        (`peer_agents`) çalıştırarak geçerli sonuçları liste halinde çekirdeğe iletir.
        """
        results: List[Dict[str, Any]] = []

        primary_result = self.investigate(query)
        if primary_result is not None:
            results.append(primary_result)

        for peer_fn in self._peer_agents:
            try:
                peer_raw = peer_fn(query)
                if peer_raw is not None:
                    normalized = self._validate_and_normalize_schema(
                        peer_raw,
                        default_source_id=str(peer_raw.get("source_id", "peer_source")),
                    )
                    if normalized is not None:
                        results.append(normalized)
            except Exception:
                continue

        return results

    # =========================================================================
    # DAHİLİ VERİ ÇIKARIMI, SOYUT İSTEM OLUŞTURMA VE ŞEMA DOĞRULAMA
    # =========================================================================

    @staticmethod
    def _extract_query_fields(query: Any) -> Dict[str, Any]:
        """
        Hem `RESEARCH_SPEC.md` şartnamesindeki alan adlarını (`anomaly_delta.difference`,
        `context_vector.values`, `candidate_dimension_name`) hem de `core.domain.TargetQuery`
        alan adlarını (`delta_state`, `active_vector.values`, `observed_outcome`)
        alan-bağımsız şekilde okur.
        """
        difference: Any = None
        anomaly_delta_obj = getattr(query, "anomaly_delta", None)
        if anomaly_delta_obj is not None:
            difference = getattr(anomaly_delta_obj, "difference", anomaly_delta_obj)
        elif hasattr(query, "delta_state"):
            difference = getattr(query, "delta_state")

        context_values: Dict[str, float] = {}
        context_vec_obj = getattr(query, "context_vector", None) or getattr(
            query, "active_vector", None
        )
        if context_vec_obj is not None:
            raw_vals = getattr(context_vec_obj, "values", context_vec_obj)
            if isinstance(raw_vals, dict):
                context_values = {
                    str(k): float(v)
                    for k, v in raw_vals.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }

        candidate_dim: Optional[str] = getattr(query, "candidate_dimension_name", None)
        if candidate_dim is not None:
            candidate_dim = str(candidate_dim).strip() or None

        observed_outcome: Any = getattr(query, "observed_outcome", None)

        return {
            "difference": difference,
            "context_values": context_values,
            "candidate_dimension_name": candidate_dim,
            "observed_outcome": observed_outcome,
        }

    @staticmethod
    def _build_investigation_prompt(payload: Dict[str, Any]) -> str:
        """
        İnvaryant #3 (Mutlak Alan Bağımsızlığı) uyarınca hiçbir spesifik alana
        atıfta bulunmayan, tamamen soyut matematiksel/epistemik sorgu istemini üretir.
        """
        prompt_data = {
            "anomaly_vector_delta": payload.get("difference"),
            "active_subspace_features": payload.get("context_values"),
            "candidate_dimension_hint": payload.get("candidate_dimension_name"),
            "observed_outcome": payload.get("observed_outcome"),
            "required_response_schema": {
                "dimension_name": "string (snake_case, e.g. latent_feature_k)",
                "statistic": "float (estimated scale or magnitude)",
                "confidence": "float in range [0.0, 1.0)",
            },
        }
        return (
            "Analyze the anomaly vector Δ, active subspace features, and return the missing "
            "latent dimension index/name (e.g. latent_feature_k) and its estimated scale.\n"
            "Return ONLY a valid JSON object matching `required_response_schema`:\n"
            f"{json.dumps(prompt_data, ensure_ascii=False)}"
        )

    def _dispatch_request(self, prompt: str, payload: Dict[str, Any]) -> Any:
        """
        Özel `transport_fn` enjekte edilmişse onu çağırır; aksi halde `api_key`
        mevcutsa standart `urllib.request` ile Gemini REST API çağrısı yapar.
        """
        if self._transport_fn is not None:
            return self._transport_fn(prompt, payload)

        if not self.api_key:
            return None

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        request_body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        encoded_body = json.dumps(request_body).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=encoded_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
            raw_http = response.read().decode("utf-8")
            http_json = json.loads(raw_http)
            candidates = http_json.get("candidates", [])
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return None
            return parts[0].get("text")

    @staticmethod
    def _parse_llm_output(raw_output: Any) -> Optional[Dict[str, Any]]:
        """
        LLM'den dönen ham metni (Markdown ```json blokları dahil) veya sözlüğü
        parse ederek Python `dict` nesnesine çevirir.
        """
        if isinstance(raw_output, dict):
            return dict(raw_output)

        if not isinstance(raw_output, str):
            return None

        cleaned = raw_output.strip()
        if not cleaned:
            return None

        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
        if fenced_match:
            cleaned = fenced_match.group(1).strip()
        else:
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned = cleaned[start_idx : end_idx + 1]

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        return parsed if isinstance(parsed, dict) else None

    def _validate_and_normalize_schema(
        self,
        data: Dict[str, Any],
        default_source_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        RESEARCH_SPEC.md Bölüm 3'te tanımlanan standart şemayı ve sınır kurallarını uygular:
        - `dimension_name`: str, `snake_case` formatına normalize edilir.
        - `statistic`: geçerli bir `float` (NaN/Inf olamaz).
        - `confidence`: `float`, [0.0, 1.0) aralığında; >= 1.0 ise `0.999` değerine kırpılır.
        - `source_id`: str.
        """
        if not isinstance(data, dict):
            return None

        raw_dim = data.get("dimension_name")
        if not isinstance(raw_dim, str) or not raw_dim.strip():
            return None

        snake_dim = self._to_snake_case(raw_dim)
        if not snake_dim:
            return None

        raw_stat = data.get("statistic")
        if (
            raw_stat is None
            or isinstance(raw_stat, bool)
            or not isinstance(raw_stat, (int, float))
        ):
            return None

        statistic_val = float(raw_stat)
        if math.isnan(statistic_val) or math.isinf(statistic_val):
            return None

        raw_conf = data.get("confidence")
        if (
            raw_conf is None
            or isinstance(raw_conf, bool)
            or not isinstance(raw_conf, (int, float))
        ):
            return None

        conf_val = float(raw_conf)
        if math.isnan(conf_val) or math.isinf(conf_val):
            return None

        if conf_val >= 1.0:
            conf_val = self.MAX_ALLOWED_CONFIDENCE
        elif conf_val < 0.0:
            conf_val = 0.0

        fallback_source = default_source_id if default_source_id else self.source_id
        raw_source = data.get("source_id", fallback_source)
        if not isinstance(raw_source, str) or not raw_source.strip():
            source_id_val = fallback_source
        else:
            source_id_val = raw_source.strip()

        return {
            "dimension_name": snake_dim,
            "statistic": statistic_val,
            "confidence": conf_val,
            "source_id": source_id_val,
        }

    @staticmethod
    def _to_snake_case(text: str) -> str:
        """
        Verilen soyut boyut adını standart `snake_case` formatına dönüştürür
        (Örn: 'LatentDim (Alpha)' -> 'latent_dim_alpha', 'DiscoveredLatentX' -> 'discovered_latent_x').
        """
        step1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text.strip())
        step2 = re.sub(r"[^a-zA-Z0-9]+", "_", step1).strip("_").lower()
        return step2


__all__ = ["GeminiResearchAgent"]
