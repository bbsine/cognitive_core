"""
STORAGE_SPEC.md & Invariant #3 - JsonMemoryRepository Kalıcılık, Re-Hydration ve Dayanıklılık Testleri.

Çalıştırma komutu:
    python -m unittest tests/unit/test_json_memory_repo.py -v
"""

from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest

from adapters.storage.json_memory_repo import JsonMemoryRepository
from core.domain import (
    ComponentVector,
    ConfidenceScore,
    DimensionAxis,
    GlobalParameterSpace,
    MicroModule,
    Observation,
)


class TestJsonMemoryRepository(unittest.TestCase):
    """JsonMemoryRepository (STORAGE_SPEC.md) alan-bağımsız doğrulama testleri."""

    def test_save_and_rehydrate_weights_and_manual_weight_modification(self) -> None:
        """
        1. Cebirsel ağırlık çıkarımı ile JSON'a yazmalı,
        2. Kullanıcı JSON dosyasındaki bir ağırlığı elle değiştirdiğinde (örn: 0.985 -> 10.0),
           load_state sonrasında motor yeni ağırlıkla tahmin üretmelidir.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            kb_path = Path(tmp_dir) / "storage" / "knowledge_base.json"
            space = GlobalParameterSpace()
            space.register_dimension(DimensionAxis(name="dim_alpha", data_type="continuous"))
            space.register_dimension(DimensionAxis(name="dim_beta", data_type="continuous"))

            repo = JsonMemoryRepository(file_path=kb_path, parameter_space=space)

            # Y = 0.985 * dim_alpha + 1.012 * dim_beta - 0.004
            mod = MicroModule(
                module_id="module_syn_01",
                active_dimensions={"dim_alpha", "dim_beta"},
                confidence=ConfidenceScore(score=0.942),
                predict_fn=lambda vec: (
                    vec.values["dim_alpha"] * 0.985
                    + vec.values["dim_beta"] * 1.012
                    - 0.004
                ),
            )
            repo.save_active_module(mod)
            repo.save_observation(
                Observation(
                    id="obs_1",
                    features=ComponentVector(values={"dim_alpha": 1.0, "dim_beta": 1.0}),
                    observed_outcome=1.993,
                )
            )

            # JSON formatının STORAGE_SPEC.md ile birebir uyumu
            raw_json = json.loads(kb_path.read_text(encoding="utf-8"))
            self.assertEqual(raw_json["version"], "1.0.0")
            self.assertEqual(len(raw_json["active_modules_h3"]), 1)
            self.assertEqual(
                raw_json["active_modules_h3"][0]["module_id"], "module_syn_01"
            )
            self.assertAlmostEqual(
                raw_json["active_modules_h3"][0]["weights"]["dim_alpha"], 0.985, places=6
            )
            self.assertAlmostEqual(
                raw_json["active_modules_h3"][0]["bias"], -0.004, places=6
            )

            # Kullanıcı JSON dosyasını açıp 'dim_alpha' ağırlığını 10.0, bias'ı 2.0 yapıyor
            raw_json["active_modules_h3"][0]["weights"]["dim_alpha"] = 10.0
            raw_json["active_modules_h3"][0]["bias"] = 2.0
            kb_path.write_text(json.dumps(raw_json, indent=2), encoding="utf-8")

            # Durumu yeniden yükle ve yeni ağırlıkla tahmin üretildiğini doğrula
            repo.load_state()
            rehydrated_mod = repo.get_active_modules()[0]
            test_vec = ComponentVector(values={"dim_alpha": 3.0, "dim_beta": 1.0})
            # Beklenen: 10.0 * 3.0 + 1.012 * 1.0 + 2.0 = 33.012
            self.assertAlmostEqual(rehydrated_mod.evaluate(test_vec), 33.012, places=6)

    def test_manual_dimension_deletion_quarantines_broken_module_without_crashing(self) -> None:
        """
        Kullanıcı JSON dosyasındaki 'global_dimensions' listesinden bir boyutu
        elle sildiğinde sistem çökmemeli; o boyuta bağımlı olan H_3 modülü
        otomatik olarak H_2 karantina havuzuna taşınmalıdır.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            kb_path = Path(tmp_dir) / "storage" / "knowledge_base.json"
            space = GlobalParameterSpace()
            space.register_dimension(DimensionAxis(name="dim_alpha", data_type="continuous"))
            space.register_dimension(DimensionAxis(name="dim_beta", data_type="continuous"))

            repo = JsonMemoryRepository(file_path=kb_path, parameter_space=space)
            mod = MicroModule(
                module_id="module_syn_01",
                active_dimensions={"dim_alpha", "dim_beta"},
                confidence=ConfidenceScore(score=0.80),
                predict_fn=lambda vec: vec.values["dim_alpha"] + vec.values["dim_beta"],
            )
            repo.save_active_module(mod)

            # Kullanıcı JSON'dan 'dim_beta' boyutunu elle siliyor
            raw_json = json.loads(kb_path.read_text(encoding="utf-8"))
            raw_json["global_dimensions"] = [
                d for d in raw_json["global_dimensions"] if d["name"] != "dim_beta"
            ]
            kb_path.write_text(json.dumps(raw_json, indent=2), encoding="utf-8")

            # Sistem çökmeden yüklenmeli ve module_syn_01 H_2 karantinasına alınmalı
            state = repo.load_state()
            self.assertEqual(len(repo.get_active_modules()), 0)
            self.assertEqual(len(repo.get_quarantined_modules()), 1)
            self.assertEqual(repo.get_quarantined_modules()[0].module_id, "module_syn_01")
            self.assertIn("module_syn_01", state["quarantined_due_to_missing_dimensions"])


if __name__ == "__main__":
    unittest.main()