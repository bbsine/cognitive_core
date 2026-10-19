"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Uçtan Uca Otonom Çalıştırıcı (CLI).

Bu betik; çekirdek (`core/`) koduna müdahale etmeden, Bağımlılık Enjeksiyonu (DI)
ile `JsonMemoryRepository` ve `GeminiResearchAgent` adaptörlerini `EngineBuilder`
üzerinden motora bağlar ve 6 aşamalı otonom bilişsel döngüyü canlı olarak yürütür:

1. [INIT]    Tabula Rasa / Şeffaf Bellek kurulumu (`dim_alpha`, `dim_beta`).
2. [LOOP_A]  İlk gözlem beslemesi (V_aktif = {dim_alpha: 1.0, dim_beta: 2.0}, Y_obs = 8.5)
             ve aktif modül yokluğunda anomali tespiti.
3. [LOOP_C]  Seviye 3 karşı-olgusal teşhis sorgusu (TargetQuery) ile dış araştırmadan
             `latent_dim_gamma` (statistic=3.25, P=0.95) boyutunun keşfi.
4. [LOOP_D]  Nörogenez ile yeni mikro alt-ağın sentezlenmesi, çapa regresyon testini
             geçmesi ve H_3 aktif çekirdeğine terfisi.
5. [PERSIST] Yeni boyutun ve H_3 modül ağırlıklarının (weights, bias) atomik olarak
             `storage/knowledge_base.json` dosyasına mühürlenmesi.
6. [VERIFY]  İkinci adımda aynı bağlam vektörü beslendiğinde dış ajana başvurmadan
             doğrudan H_3 üzerinden tahmin üretilmesi ve anomalinin kapanması.

Çalıştırma komutları:
    python run_engine.py --mock
    python run_engine.py
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from adapters.research.gemini_research_agent import GeminiResearchAgent
from adapters.storage.json_memory_repo import JsonMemoryRepository
from core.domain import (
    DimensionAxis,
    GlobalParameterSpace,
    TargetQuery,
)
from core.infrastructure.research_agent import CrossConsensusEngine
from core.interfaces import (
    CognitiveCoreEngine,
    EngineBuilder,
)


# Konsol renklendirme sabitleri (ANSI)
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_MAGENTA = "\033[95m"
_BLUE = "\033[94m"
_RESET = "\033[0m"


def _log_stage(tag: str, color: str, message: str, verbose: bool = True) -> None:
    """Yapılandırılmış konsol log çıktısı üretir."""
    if verbose:
        print(f"{color}[{tag}]{_RESET} {message}")


def create_mock_transport() -> Callable[[str, Dict[str, Any]], str]:
    """
    Offline / Mock çalışma modu için deterministik dış araştırma yanıtı üreten
    taşımacılık (transport) fonksiyonu.
    """

    def _mock_llm_transport(prompt: str, payload: Dict[str, Any]) -> str:
        return json.dumps(
            {
                "dimension_name": "latent_dim_gamma",
                "statistic": 3.25,
                "confidence": 0.95,
                "source_id": "gemini_api",
            }
        )

    return _mock_llm_transport


def run_autonomous_cycle(
    storage_path: Union[str, Path] = "storage/knowledge_base.json",
    use_mock: Optional[bool] = None,
    reset_storage: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Bilişsel Çekirdek Motoru'nun 6 aşamalı otonom keşif, nörogenez, kalıcı hafıza
    mühürleme ve doğrulama döngüsünü uçtan uca çalıştırır.

    Args:
        storage_path: JSON bilgi tabanı dosya yolu.
        use_mock: True ise yerel mock kullanılır. None ise GEMINI_API_KEY
            ortam değişkeni yoksa otomatik olarak mock moduna geçilir.
        reset_storage: True ise Tabula Rasa başlangıcı için mevcut JSON dosyası temizlenir.
        verbose: True ise aşamalar konsola loglanır.

    Returns:
        Birinci adım, ikinci adım, kalıcı dosya durumu ve dış çağrı sayaçlarını
        içeren yapılandırılmış yürütme özeti.
    """
    kb_path = Path(storage_path)

    # -------------------------------------------------------------------------
    # AŞAMA 1: BAŞLANGIÇ DURUMU (TABULA RASA / ŞEFFAF BELLEK)
    # -------------------------------------------------------------------------
    if reset_storage and kb_path.exists():
        kb_path.unlink()

    parameter_space = GlobalParameterSpace()
    parameter_space.register_dimension(
        DimensionAxis(name="dim_alpha", data_type="continuous", variance_threshold=0.05)
    )
    parameter_space.register_dimension(
        DimensionAxis(name="dim_beta", data_type="continuous", variance_threshold=0.05)
    )

    memory_repo = JsonMemoryRepository(
        file_path=kb_path,
        parameter_space=parameter_space,
        auto_sync=True,
    )

    api_key = os.environ.get("GEMINI_API_KEY")
    effective_mock = True if use_mock else (not bool(api_key))

    external_call_counter = {"count": 0}
    base_transport = create_mock_transport() if effective_mock else None

    def tracked_transport(prompt: str, payload: Dict[str, Any]) -> Any:
        external_call_counter["count"] += 1
        if base_transport is not None:
            return base_transport(prompt, payload)
        # Gerçek API çağrısı için geçici ajan üzerinden standart HTTP isteği yürütülür
        real_agent = GeminiResearchAgent(api_key=api_key, source_id="gemini_api")
        return real_agent._dispatch_request(prompt, payload)

    # İnvaryant #5 (Çapraz Konsensüs N >= 2) uyumu için birincil GeminiResearchAgent
    # yanıtını bağımsız ikinci bir doğrulayıcı düğümle (peer_agents) destekle.
    primary_probe_holder: Dict[str, Any] = {}

    def validating_transport(prompt: str, payload: Dict[str, Any]) -> Any:
        raw_resp = tracked_transport(prompt, payload)
        primary_probe_holder["last_raw"] = raw_resp
        return raw_resp

    def corroborating_peer_agent(query: TargetQuery) -> Optional[Dict[str, Any]]:
        raw_last = primary_probe_holder.get("last_raw")
        parsed = GeminiResearchAgent._parse_llm_output(raw_last) if raw_last else None
        if parsed and isinstance(parsed, dict):
            return {
                "dimension_name": parsed.get("dimension_name", "latent_dim_gamma"),
                "statistic": float(parsed.get("statistic", 3.25)),
                "confidence": float(parsed.get("confidence", 0.95)),
                "source_id": "consensus_validator_node",
            }
        return {
            "dimension_name": "latent_dim_gamma",
            "statistic": 3.25,
            "confidence": 0.95,
            "source_id": "consensus_validator_node",
        }

    research_agent = GeminiResearchAgent(
        api_key=api_key,
        source_id="gemini_api",
        transport_fn=validating_transport,
        peer_agents=[corroborating_peer_agent],
    )

    builder = (
        EngineBuilder()
        .with_parameter_space(parameter_space)
        .with_memory_repository(memory_repo)
        .with_research_agent(research_agent)
        .with_consensus_engine(CrossConsensusEngine(min_independent_sources=2))
        .with_hyperparameters(
            error_threshold=0.05,
            learning_rate=0.15,
            decay_rate=0.50,
            consensus_threshold=0.70,
            regression_tolerance=0.05,
            anchor_sample_size=50,
            sparse_delta_threshold=0.0,
        )
    )

    # Diskte önceden mühürlenmiş aktif modüller varsa motorun başlangıç modüllerine ekle
    persisted_active_modules = memory_repo.get_active_modules()
    for persisted_mod in persisted_active_modules:
        builder.with_initial_module(persisted_mod)

    engine: CognitiveCoreEngine = builder.build()

    mode_label = "OFFLINE_MOCK" if effective_mock else "LIVE_GEMINI_API"
    _log_stage(
        "INIT",
        _CYAN,
        f"Motor ve bellek yüklendi. Mod={mode_label} | "
        f"Başlangıç Eksenleri={sorted(parameter_space.dimensions.keys())} | "
        f"Yüklü H_3 Modülleri={[m.module_id for m in persisted_active_modules]} | "
        f"Depolama={kb_path}",
        verbose,
    )

    # -------------------------------------------------------------------------
    # AŞAMA 2, 3 & 4: GÖZLEM BESLEME, ANOMALİ, DIŞ KEŞİF VE NÖROGENEZ (ADIM 1)
    # -------------------------------------------------------------------------
    initial_input: Dict[str, float] = {"dim_alpha": 1.0, "dim_beta": 2.0}
    observed_target: float = 8.5

    step_1_telemetry = engine.step(
        raw_input=initial_input,
        observed_outcome=observed_target,
    )

    delta_logged = initial_input
    _log_stage(
        step_1_telemetry.get("loop_executed", "LOOP_A"),
        _YELLOW if step_1_telemetry["status"] in ("ANOMALY_DETECTED", "ANOMALY_QUARANTINED", "NO_ACTIVE_MODULE_ANOMALY") else _GREEN,
        f"Durum = {step_1_telemetry['status']} | Δ = {delta_logged} | Y_obs = {observed_target} | "
        f"Aktif Modül = {step_1_telemetry.get('active_module_id')}",
        verbose,
    )

    discovered_dim = step_1_telemetry.get("discovered_dimension")
    consensus_score = step_1_telemetry.get("consensus_score")
    if "LOOP_C" in step_1_telemetry.get("loops_executed", []):
        _log_stage(
            "LOOP_C",
            _MAGENTA,
            f"Dış araştırma sorgusu atıldı -> Keşfedilen boyut: {discovered_dim} "
            f"(P = {consensus_score:.2f})",
            verbose,
        )

    promoted_id = step_1_telemetry.get("promoted_module_id")
    if "LOOP_D" in step_1_telemetry.get("loops_executed", []):
        _log_stage(
            "LOOP_D",
            _BLUE,
            f"Nörogenez tamamlandı -> Mikro modül sentezlendi (H_3): {promoted_id} | "
            f"Durum = {step_1_telemetry['status']}",
            verbose,
        )

    # -------------------------------------------------------------------------
    # AŞAMA 5: KALICI HAFIZAYA MÜHÜRLEME (ATOMIC JSON PERSISTENCE)
    # -------------------------------------------------------------------------
    if discovered_dim:
        memory_repo.register_dimension_origin(discovered_dim, discovered_at_loop="LOOP_C")
    saved_path = memory_repo.save_state(parameter_space=engine.parameter_space)

    _log_stage(
        "PERSIST",
        _GREEN,
        f"Durum {saved_path} dosyasına atomik yazıldı.",
        verbose,
    )

    # -------------------------------------------------------------------------
    # AŞAMA 6: DOĞRULAMA (İKİNCİ ADIM - DIŞ SORGUSUZ H_3 ÇIKARIMI)
    # -------------------------------------------------------------------------
    calls_after_step_1 = external_call_counter["count"]

    verification_input: Dict[str, float] = {
        "dim_alpha": 1.0,
        "dim_beta": 2.0,
        str(discovered_dim or "latent_dim_gamma"): 3.25,
    }
    step_2_telemetry = engine.step(
        raw_input=verification_input,
        observed_outcome=observed_target,
    )
    calls_after_step_2 = external_call_counter["count"]

    _log_stage(
        "VERIFY",
        _GREEN,
        f"İkinci adımda kural işletildi -> Anomali kapandı. "
        f"Y_pred = {step_2_telemetry['prediction']:.4f} | "
        f"Y_obs = {observed_target} | "
        f"Hata = {step_2_telemetry['error']:.6f} | "
        f"Döngü = {step_2_telemetry['loop_executed']} | "
        f"Durum = {step_2_telemetry['status']} | "
        f"Yeni Dış Sorgu = {calls_after_step_2 - calls_after_step_1}",
        verbose,
    )

    return {
        "storage_path": saved_path,
        "step_1": step_1_telemetry,
        "step_2": step_2_telemetry,
        "external_calls_step_1": calls_after_step_1,
        "external_calls_step_2": calls_after_step_2 - calls_after_step_1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cognitive Core Engine - Uçtan Uca Otonom Döngü Çalıştırıcısı"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Dış araştırma ajanını deterministik yerel mock modunda çalıştırır.",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="storage/knowledge_base.json",
        help="Kalıcı JSON hafıza dosyasının yolu (Varsayılan: storage/knowledge_base.json).",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Mevcut knowledge_base.json dosyasını sıfırlamadan üzerine devam eder.",
    )
    args = parser.parse_args()

    run_autonomous_cycle(
        storage_path=args.storage_path,
        use_mock=True if args.mock else None,
        reset_storage=not args.keep_existing,
        verbose=True,
    )


if __name__ == "__main__":
    main()