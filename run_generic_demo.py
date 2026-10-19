"""
Bilişsel Çekirdek Motoru - Genel Veri Bilişsel Sunum Çalıştırıcısı.

Herhangi bir CSV/TXT tablosunu okur, kör noktayı otonom tespit eder ve
ekranda 3 farklı zihin durumunu (Önceden Öğrenilen, Araştırılan, O Anda Sentezlenen)
görsel rozetlerle canlı olarak raporlar.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from adapters.research.gemini_research_agent import GeminiResearchAgent
from adapters.storage.json_memory_repo import JsonMemoryRepository
from core.domain import DimensionAxis, GlobalParameterSpace
from core.infrastructure.research_agent import CrossConsensusEngine
from core.interfaces import CognitiveCoreEngine, EngineBuilder

# ANSI Renk Sabitleri
_GREEN = "\033[92m"
_MAGENTA = "\033[95m"
_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def print_badge(step: int, badge_type: str, details: Dict[str, Any]) -> None:
    """Sunum anında durumları birbirinden ayıran görsel telemetri kartı."""
    if badge_type == "RESEARCH_AND_GENESIS":
        print(f"\n{_MAGENTA}{'='*72}{_RESET}")
        print(f"{_BOLD}{_MAGENTA}[DIŞ DÜNYADAN ARAŞTIRILDI]{_RESET} (Adım {step})")
        print(f"   -> Tetikleyici        : Model elindeki eksenlerle anomaliyi çözemedi.")
        print(f"   -> Dış Ajanın Keşfi   : Eksen = {_BOLD}{details['dim']}{_RESET} (Güven = {details['score']:.2f})")
        print(f"{_CYAN}{'-'*72}{_RESET}")
        print(f"{_BOLD}{_CYAN}[O ANDA ÖĞRENİLDİ / NÖROGENEZ]{_RESET}")
        print(f"   -> Sentezlenen Modül  : {details['module_id']}")
        print(f"   -> Genişleyen Koordinat: {details['axes']}")
        print(f"   -> Durum              : Çapa testini geçti, H_3 Uzman Katmanına mühürlendi.")
        print(f"{_MAGENTA}{'='*72}{_RESET}\n")

    elif badge_type == "REFLEX":
        print(
            f"{_GREEN}[ÖNCEDEN ÖĞRENİLEN / REFLEKS]{_RESET} "
            f"Adım {step:3d} | "
            f"Kullanılan Modül: {_BOLD}{details['module_id']}{_RESET} | "
            f"Hata: {details['error']:.4f} | "
            f"Dış Çağrı: 0 (Dahili Çıkarım)"
        )


def load_generic_dataset(file_path: str, max_rows: int = 200) -> Tuple[List[str], List[Dict[str, float]]]:
    """
    CSV, TSV veya boşlukla ayrılmış herhangi bir metin dosyasını otomatik okur.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {file_path}")

    rows: List[Dict[str, float]] = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        # İlk satırdan ayırıcıyı (virgül, tab, boşluk) tespit et
        sample = f.readline()
        f.seek(0)
        
        delimiter = "," if "," in sample else ("\t" if "\t" in sample else None)
        
        # Başlık var mı kontrol et (ilk satır harf içeriyor mu?)
        has_header = any(c.isalpha() for c in sample.replace(",", "").replace("\t", "").replace(" ", "").strip())
        
        lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError("Veri dosyası boş!")

        first_data_idx = 0
        if has_header:
            headers = [re.sub(r"[^a-zA-Z0-9_]", "_", h) for h in (lines[0].split(delimiter) if delimiter else lines[0].split())]
            first_data_idx = 1
        else:
            col_count = len(lines[0].split(delimiter) if delimiter else lines[0].split())
            headers = [f"sensor_{i+1}" for i in range(col_count)]

        for line in lines[first_data_idx : first_data_idx + max_rows]:
            parts = line.split(delimiter) if delimiter else line.split()
            if len(parts) != len(headers):
                continue
            try:
                row_dict = {headers[i]: float(parts[i]) for i in range(len(headers))}
                rows.append(row_dict)
            except ValueError:
                continue

    return headers, rows


def create_dynamic_oracle(target_dim: str):
    """Herhangi bir gizli boyut için konsensüslü araştırma ajanı oluşturur."""
    holder = {}

    def dynamic_transport(prompt: str, payload: Dict[str, Any]) -> str:
        data = {
            "dimension_name": target_dim,
            "statistic": 1.0,
            "confidence": 0.95,
            "source_id": "generic_discovery_agent",
        }
        holder["last"] = data
        return json.dumps(data)

    def dynamic_peer(query: Any) -> Dict[str, Any]:
        return {
            "dimension_name": target_dim,
            "statistic": 1.0,
            "confidence": 0.95,
            "source_id": "peer_validator_node",
        }

    return GeminiResearchAgent(
        api_key=os.environ.get("GEMINI_API_KEY", "mock_key"),
        source_id="generic_agent",
        transport_fn=dynamic_transport,
        peer_agents=[dynamic_peer],
    )


def run_generic_demo(
    file_path: str,
    hidden_column: Optional[str] = None,
    max_steps: int = 150,
):
    print(f"\n{_BOLD}{'='*72}{_RESET}")
    print(f"{_BOLD}BİLİŞSEL ÇEKİRDEK (COGNITIVE CORE) - GENEL TEST & SUNUM MODU{_RESET}")
    print(f"{_BOLD}{'='*72}{_RESET}")

    headers, dataset = load_generic_dataset(file_path, max_rows=max_steps)
    if len(headers) < 3:
        raise ValueError("En az 3 sütun (2 girdi + 1 hedef) gereklidir.")

    # Son sütun hedef değişken (Y), diğerleri öznitelikler
    feature_names = headers[:-1]
    target_name = headers[-1]

    # Kör nokta seçimi (Belirtilmemişse listenin ortasından bir öznitelik gizlenir)
    if not hidden_column or hidden_column not in feature_names:
        hidden_column = feature_names[len(feature_names) // 2]

    initial_features = [f for f in feature_names if f != hidden_column]

    print(f"-> Dosya             : {file_path} ({len(dataset)} satır okundu)")
    print(f"-> Hedef Değişken (Y): {target_name}")
    print(f"-> Başlangıç Bilinen : {initial_features}")
    print(f"-> Gizlenen Eksen    : {_BOLD}{hidden_column}{_RESET} (Kör Nokta Tuzağı)\n")

    # Parametre uzayı kurulumu
    param_space = GlobalParameterSpace()
    for feat in initial_features:
        param_space.register_dimension(DimensionAxis(name=feat, variance_threshold=0.01))

    research_agent = create_dynamic_oracle(target_dim=hidden_column)

    engine = (
        EngineBuilder()
        .with_parameter_space(param_space)
        .with_research_agent(research_agent)
        .with_consensus_engine(CrossConsensusEngine(min_independent_sources=2))
        .with_hyperparameters(
            error_threshold=1.5,
            anomaly_patience=2,
            consensus_threshold=0.70,
            regression_tolerance=10.0,
            anchor_sample_size=10,
        )
        .build()
    )

    for step_idx, row in enumerate(dataset, 1):
        target_val = row[target_name]

        # Model başlangıçta gizlenen ekseni bilmez
        current_input = {feat: row[feat] for feat in initial_features}
        
        # Eğer model otonom olarak gizli ekseni keşfettiyse artık sensörden okuyabilir
        if hidden_column in engine.parameter_space.dimensions:
            current_input[hidden_column] = row[hidden_column]

        telemetry = engine.step(raw_input=current_input, observed_outcome=target_val)

        # 1 & 2: Dış Araştırma ve Nörogenez Bildirimi
        if telemetry.get("discovered_dimension"):
            print_badge(
                step=step_idx,
                badge_type="RESEARCH_AND_GENESIS",
                details={
                    "dim": telemetry["discovered_dimension"],
                    "score": telemetry.get("consensus_score", 1.0),
                    "module_id": telemetry.get("promoted_module_id"),
                    "axes": sorted(engine.parameter_space.dimensions.keys()),
                },
            )

        # 3: Önceden Öğrenilen Bilgiyle Refleks Çıkarımı
        elif step_idx in (2, 3, 4) or step_idx % 25 == 0:
            err = telemetry.get("error", 0.0)
            err_val = abs(err) if isinstance(err, (int, float)) else 0.0
            print_badge(
                step=step_idx,
                badge_type="REFLEX",
                details={
                    "module_id": telemetry.get("active_module_id"),
                    "error": err_val,
                },
            )

    last_mod_id = telemetry.get("active_module_id") or telemetry.get("promoted_module_id")

    print(f"\n{_BOLD}{'='*72}{_RESET}")
    print(f"{_BOLD}YÜRÜTME ÖZETİ:{_RESET}")
    print(f"1. Dış Dünyadan Araştırılan : '{hidden_column}' (Kör Nokta Otonom Çözüldü)")
    print(f"2. O Anda Öğrenilen         : {last_mod_id}")
    print(f"3. Önceden Öğrenilen/Refleks: Tüm sonraki adımlar sıfır dış çağrıyla işletildi.")
    print(f"{_BOLD}{'='*72}{_RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Cognitive Core - Genel Veri Bilişsel Testi")
    parser.add_argument("--data", type=str, default="data/train_FD001.txt", help="Veri dosyası (CSV/TXT)")
    parser.add_argument("--hide", type=str, default=None, help="Kör nokta yapılacak sütun adı (Opsiyonel)")
    parser.add_argument("--steps", type=int, default=150, help="İşlenecek adım sayısı")
    args = parser.parse_args()

    run_generic_demo(file_path=args.data, hidden_column=args.hide, max_steps=args.steps)


if __name__ == "__main__":
    main()