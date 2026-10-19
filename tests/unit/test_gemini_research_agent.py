"""
RESEARCH_SPEC.md & Invariant #3 - GeminiResearchAgent Protokol, Şema, Kırpma (P < 1.0)
ve Mutlak Alan Bağımsızlığı Denetim Testleri.

Çalıştırma komutu:
    python -m unittest tests/unit/test_gemini_research_agent.py -v
"""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import unittest
import urllib.error

from adapters.research.gemini_research_agent import GeminiResearchAgent
from core.domain import (
    ComponentVector,
    TargetQuery,
)


@dataclass(frozen=True)
class _SpecAnomalyDelta:
    difference: float


@dataclass(frozen=True)
class _SpecContextVector:
    values: Dict[str, float]


@dataclass(frozen=True)
class _SpecTargetQuery:
    anomaly_delta: _SpecAnomalyDelta
    context_vector: _SpecContextVector
    candidate_dimension_name: Optional[str] = None


class TestGeminiResearchAgent(unittest.TestCase):
    """
    GeminiResearchAgent (RESEARCH_SPEC.md) birim, mock ve İnvaryant #3 denetim testleri.
    """

    def setUp(self) -> None:
        self.core_query = TargetQuery(
            delta_state={"feature_dim_0": 0.0, "feature_dim_1": 1.0},
            active_vector=ComponentVector(values={"feature_dim_0": 1.0, "feature_dim_1": 1.0}),
            observed_outcome=1.25,
        )
        self.spec_query = _SpecTargetQuery(
            anomaly_delta=_SpecAnomalyDelta(difference=1.25),
            context_vector=_SpecContextVector(values={"feature_dim_0": 1.0, "feature_dim_1": 1.0}),
            candidate_dimension_name="LatentDimAlpha",
        )

    def test_investigate_returns_exact_standard_schema_and_normalizes_snake_case(self) -> None:
        """
        Adaptör, LLM çıktısını parse ederek RESEARCH_SPEC.md Bölüm 3'teki 4 zorunlu
        anahtarı ('dimension_name', 'statistic', 'confidence', 'source_id') dönmeli,
        boyut adını snake_case formatına çevirmeli ve soyut istem (prompt) üretmelidir.
        """
        captured_payload: Dict[str, Any] = {}
        captured_prompt: str = ""

        def mock_transport(prompt: str, payload: Dict[str, Any]) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            captured_payload.update(payload)
            return json.dumps(
                {
                    "dimension_name": "LatentDim (Alpha)",
                    "statistic": 1.25,
                    "confidence": 0.95,
                }
            )

        agent = GeminiResearchAgent(
            source_id="gemini_api",
            transport_fn=mock_transport,
        )

        result = agent.investigate(self.spec_query)  # type: ignore[arg-type]

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            set(result.keys()),
            {"dimension_name", "statistic", "confidence", "source_id"},
        )
        self.assertEqual(result["dimension_name"], "latent_dim_alpha")
        self.assertAlmostEqual(result["statistic"], 1.25)
        self.assertAlmostEqual(result["confidence"], 0.95)
        self.assertEqual(result["source_id"], "gemini_api")

        # Prompt ve payload'un tamamen soyut (domain-agnostic) olduğunu doğrula
        self.assertIn(
            "Analyze the anomaly vector Δ, active subspace features, and return the missing "
            "latent dimension index/name (e.g. latent_feature_k) and its estimated scale.",
            captured_prompt,
        )
        self.assertAlmostEqual(captured_payload["difference"], 1.25)
        self.assertEqual(
            captured_payload["context_values"],
            {"feature_dim_0": 1.0, "feature_dim_1": 1.0},
        )
        self.assertEqual(captured_payload["candidate_dimension_name"], "LatentDimAlpha")

    def test_confidence_clipping_enforces_invariant_1_when_llm_returns_one_or_above(self) -> None:
        """
        Sınır Kuralı (Invariant #1): Model confidence olarak 1.0 veya > 1.0 dönerse,
        adaptör bunu kesinlikle 0.999 değerine kırpmalıdır.
        """
        for raw_conf in (1.0, 1.0001, 1.5, 100.0):
            with self.subTest(raw_confidence=raw_conf):
                agent = GeminiResearchAgent(
                    transport_fn=lambda p, q, c=raw_conf: (
                        f"```json\n"
                        f'{{"dimension_name": "discovered_latent_x", "statistic": 0.75, "confidence": {c}}}\n'
                        f"```"
                    )
                )
                res = agent.investigate(self.core_query)
                self.assertIsNotNone(res)
                assert res is not None
                self.assertAlmostEqual(res["confidence"], 0.999, places=6)
                self.assertLess(res["confidence"], 1.0)

    def test_resilience_returns_none_on_network_error_or_malformed_json(self) -> None:
        """
        Sınır Kuralı (Dayanıklılık): Ağ hatası (URLError/Timeout), bozuk JSON,
        eksik anahtar veya NaN değer geldiğinde adaptör sistemi çökertmemeli,
        güvenli şekilde None dönmelidir.
        """
        def failing_network_transport(prompt: str, payload: Dict[str, Any]) -> str:
            raise urllib.error.URLError("Simulated network disconnect")

        net_agent = GeminiResearchAgent(transport_fn=failing_network_transport)
        self.assertIsNone(net_agent.investigate(self.core_query))
        self.assertEqual(net_agent.query_external_sources(self.core_query), [])

        bad_json_agent = GeminiResearchAgent(
            transport_fn=lambda p, q: "{not_valid_json: 123"
        )
        self.assertIsNone(bad_json_agent.investigate(self.core_query))

        missing_key_agent = GeminiResearchAgent(
            transport_fn=lambda p, q: json.dumps(
                {"dimension_name": "latent_dim_alpha", "confidence": 0.85}
            )
        )
        self.assertIsNone(missing_key_agent.investigate(self.core_query))

        nan_agent = GeminiResearchAgent(
            transport_fn=lambda p, q: {
                "dimension_name": "latent_dim_alpha",
                "statistic": float("nan"),
                "confidence": 0.85,
            }
        )
        self.assertIsNone(nan_agent.investigate(self.core_query))

    def test_iresearch_agent_contract_query_external_sources_multi_source_support(self) -> None:
        """
        query_external_sources metodu hem birincil Gemini ajanını hem de kayıtlı
        peer ajanlarını çalıştırarak Döngü C konsensüsü için liste dönmelidir.
        """
        agent = GeminiResearchAgent(
            source_id="gemini_primary",
            transport_fn=lambda p, q: {
                "dimension_name": "latent_dim_alpha",
                "statistic": 1.25,
                "confidence": 1.0,
            },
            peer_agents=[
                lambda q: {
                    "dimension_name": "latent_dim_alpha",
                    "statistic": 1.28,
                    "confidence": 0.92,
                    "source_id": "secondary_statistical_index",
                }
            ],
        )

        responses = agent.query_external_sources(self.core_query)
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0]["source_id"], "gemini_primary")
        self.assertAlmostEqual(responses[0]["confidence"], 0.999)
        self.assertEqual(responses[1]["source_id"], "secondary_statistical_index")
        self.assertAlmostEqual(responses[1]["confidence"], 0.92)

    def test_invariant_3_codebase_domain_agnosticism_audit(self) -> None:
        """
        İnvaryant #3 Denetimi:
        core/, adapters/ ve tests/ dizinlerindeki tüm Python dosyalarını tarayarak
        alana özgü yasaklı terimlerin bulunmadığını doğrular.
        """
        project_root = Path(__file__).resolve().parents[2]
        target_dirs = [
            project_root / "core",
            project_root / "adapters",
            project_root / "tests",
        ]

        # Test dosyasının kendi içinde ham metin aramasına takılmaması için
        # yasaklı terimler parçalı birleştirme ile oluşturulur.
        forbidden_tokens: List[str] = [
            "color_" + "blue",
            "color_" + "red",
            "mass_" + "kg",
            "Mass" + "Kg",
            "gravity_" + "field",
            "Gravity" + "Field",
            "m_syn_gravity" + "_rule",
            "rel" + "eased",
            "phy" + "sics",
            "pri" + "ce",
            "9." + "81",
        ]
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(tok) for tok in forbidden_tokens) + r")\b"
        )

        violations: List[str] = []
        for folder in target_dirs:
            if not folder.exists():
                continue
            for py_file in folder.rglob("*.py"):
                content = py_file.read_text(encoding="utf-8")
                matches = pattern.findall(content)
                if matches:
                    rel_path = py_file.relative_to(project_root)
                    violations.append(f"{rel_path}: {sorted(set(matches))}")

        self.assertEqual(
            len(violations),
            0,
            f"İnvaryant #3 İhlali! Kod tabanında alana özgü terimler bulundu: {violations}",
        )


if __name__ == "__main__":
    unittest.main()