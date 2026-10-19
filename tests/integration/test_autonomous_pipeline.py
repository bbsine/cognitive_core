"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Uçtan Uca Otonom Pipeline Entegrasyon Testi.

Bu test modülü; çekirdek (`core/`) koduna dokunmadan `JsonMemoryRepository`,
`GeminiResearchAgent` ve `CognitiveCoreEngine` bileşenlerinin tam otonom
yaşam döngüsünü doğrular:
1. Tabula Rasa başlangıcı (`dim_alpha`, `dim_beta`).
2. Birinci adımda anomali tespiti, dış ajanın tetiklenmesi, `latent_dim_gamma`
   boyutunun keşfi ve H_3'e yeni mikro modül terfisi (`PROMOTED_TO_H3`).
3. `knowledge_base.json` dosyasında yeni boyutun ve modül ağırlıklarının (`weights`, `bias`)
   atomik olarak belirmesi.
4. İkinci adımda dış ajana hiç başvurmadan (`external_calls == 0`) H_3 modülü
   üzerinden doğrudan doğru tahmin üretilmesi (`REINFORCED`, `LOOP_A`).
5. Soğuk başlangıçta (Cold Restart) `knowledge_base.json` dosyasından Re-Hydration
   yapılarak dış sorgusuz tahmin üretilmesi.
6. İnvaryant #3 (Mutlak Alan Bağımsızlığı) denetiminden tüm proje genelinde 0 ihlalle geçilmesi.

Çalıştırma komutu:
    python -m unittest tests/integration/test_autonomous_pipeline.py -v
"""

from __future__ import annotations
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List
import unittest

from adapters.research.gemini_research_agent import GeminiResearchAgent
from adapters.storage.json_memory_repo import JsonMemoryRepository
from core.domain import (
    DimensionAxis,
    GlobalParameterSpace,
)
from core.infrastructure.research_agent import CrossConsensusEngine
from core.interfaces import EngineBuilder
from run_engine import run_autonomous_cycle


class TestAutonomousPipeline(unittest.TestCase):
    """
    Uçtan uca otonom keşif, nörogenez, JSON kalıcılık ve doğrulama entegrasyon testleri.
    """

    def test_autonomous_discovery_neurogenesis_persistence_and_second_step_verification(self) -> None:
        """
        1. Adımda dış ajanı tetikleyip 'latent_dim_gamma' boyutunu öğrenerek H_3'e
        yeni modül derlemeli, 'knowledge_base.json' dosyasına ağırlıklarıyla yazmalı ve
        2. Adımda dış ajana hiç başvurmadan H_3 modülüyle doğru tahmin üretmelidir.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            kb_path = Path(tmp_dir) / "storage" / "knowledge_base.json"

            summary = run_autonomous_cycle(
                storage_path=kb_path,
                use_mock=True,
                reset_storage=True,
                verbose=False,
            )

            step_1 = summary["step_1"]
            step_2 = summary["step_2"]

            # 1. Adım Doğrulamaları (Anomali -> Döngü B -> Döngü C -> Döngü D -> H_3 Terfisi)
            self.assertEqual(step_1["status"], "PROMOTED_TO_H3")
            self.assertEqual(step_1["loop_executed"], "LOOP_D")
            self.assertEqual(
                step_1["loops_executed"],
                ["LOOP_A", "LOOP_B", "LOOP_C", "LOOP_D"],
            )
            self.assertEqual(step_1["discovered_dimension"], "latent_dim_gamma")
            self.assertIsNotNone(step_1["promoted_module_id"])
            self.assertEqual(summary["external_calls_step_1"], 1)

            # Kalıcı JSON Dosyası (knowledge_base.json) Doğrulamaları
            self.assertTrue(kb_path.exists())
            persisted_data: Dict[str, Any] = json.loads(kb_path.read_text(encoding="utf-8"))

            self.assertEqual(persisted_data["version"], "1.0.0")
            dim_map = {
                d["name"]: d["discovered_at_loop"]
                for d in persisted_data["global_dimensions"]
            }
            self.assertIn("dim_alpha", dim_map)
            self.assertIn("dim_beta", dim_map)
            self.assertIn("latent_dim_gamma", dim_map)
            self.assertEqual(dim_map["latent_dim_gamma"], "LOOP_C")

            active_h3_list = persisted_data["active_modules_h3"]
            self.assertEqual(len(active_h3_list), 1)
            persisted_mod = active_h3_list[0]
            self.assertEqual(persisted_mod["module_id"], step_1["promoted_module_id"])
            self.assertEqual(
                set(persisted_mod["active_dimensions"]),
                {"dim_alpha", "dim_beta", "latent_dim_gamma"},
            )
            self.assertEqual(
                set(persisted_mod["weights"].keys()),
                {"dim_alpha", "dim_beta", "latent_dim_gamma"},
            )
            self.assertGreater(persisted_mod["weights"]["latent_dim_gamma"], 0.0)
            self.assertEqual(persisted_mod["status"], "ACTIVE")

            # 2. Adım Doğrulamaları (Dış Ajan Çağrılmadan H_3 Üzerinden Doğrudan Tahmin)
            self.assertEqual(summary["external_calls_step_2"], 0)
            self.assertEqual(step_2["status"], "REINFORCED")
            self.assertEqual(step_2["loop_executed"], "LOOP_A")
            self.assertEqual(step_2["loops_executed"], ["LOOP_A"])
            self.assertEqual(step_2["active_module_id"], step_1["promoted_module_id"])
            self.assertAlmostEqual(step_2["prediction"], 8.5, places=4)
            self.assertAlmostEqual(step_2["error"], 0.0, places=4)

    def test_cold_restart_rehydration_from_knowledge_base_closes_anomaly_without_external_agent(self) -> None:
        """
        Motor kapatılıp diskteki 'knowledge_base.json' dosyasından sıfırdan (Cold Start)
        geri yüklendiğinde (Re-Hydration), dış araştırma ajanına hiç başvurmadan
        sentezlenmiş ağırlıklarla doğrudan doğru tahmini üretmelidir.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            kb_path = Path(tmp_dir) / "storage" / "knowledge_base.json"

            # Önce otonom döngüyü çalıştırıp bilgi tabanını diske mühürle
            run_autonomous_cycle(
                storage_path=kb_path,
                use_mock=True,
                reset_storage=True,
                verbose=False,
            )

            # Şimdi yeni bir bellek adaptörü ve motor örneği oluşturup diskten yükle
            rehydrated_space = GlobalParameterSpace()
            rehydrated_repo = JsonMemoryRepository(
                file_path=kb_path,
                parameter_space=rehydrated_space,
                auto_sync=True,
            )

            # Dış ajan çağrılırsa testi başarısız kılacak tuzak transport
            external_triggered = False

            def forbidden_transport(prompt: str, payload: Dict[str, Any]) -> str:
                nonlocal external_triggered
                external_triggered = True
                raise AssertionError("Re-hydration sonrasında dış araştırma ajanı çağrılmamalıdır!")

            trap_agent = GeminiResearchAgent(transport_fn=forbidden_transport)

            restarted_engine = (
                EngineBuilder()
                .with_parameter_space(rehydrated_space)
                .with_memory_repository(rehydrated_repo)
                .with_research_agent(trap_agent)
                .with_consensus_engine(CrossConsensusEngine(min_independent_sources=2))
                .with_hyperparameters(error_threshold=0.05)
                .build()
            )

            telemetry = restarted_engine.step(
                raw_input={"dim_alpha": 1.0, "dim_beta": 2.0, "latent_dim_gamma": 3.25},
                observed_outcome=8.5,
            )

            self.assertFalse(external_triggered)
            self.assertEqual(telemetry["status"], "REINFORCED")
            self.assertEqual(telemetry["loop_executed"], "LOOP_A")
            self.assertAlmostEqual(telemetry["prediction"], 8.5, places=4)
            self.assertLessEqual(telemetry["error"], 0.05)

    def test_invariant_3_full_repository_domain_agnosticism_audit(self) -> None:
        """
        İnvaryant #3 Denetimi:
        core/, adapters/, tests/ dizinlerini ve kök dizindeki run_engine.py betiğini
        tarayarak alana özgü hiçbir terimin bulunmadığını doğrular.
        """
        project_root = Path(__file__).resolve().parents[2]
        files_to_scan: List[Path] = []

        for folder_name in ("core", "adapters", "tests"):
            folder = project_root / folder_name
            if folder.exists():
                files_to_scan.extend(folder.rglob("*.py"))

        run_script = project_root / "run_engine.py"
        if run_script.exists():
            files_to_scan.append(run_script)

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
        for py_file in files_to_scan:
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