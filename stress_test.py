import os
import json
import time
import random
import math
from core.domain import DimensionAxis, GlobalParameterSpace
from core.infrastructure.research_agent import CrossConsensusEngine
from core.interfaces import EngineBuilder
from adapters.research.gemini_research_agent import GeminiResearchAgent

# 1. Başlangıç Parametre Uzayı
param_space = GlobalParameterSpace()
param_space.register_dimension(
    DimensionAxis(name="engine_rpm", data_type="continuous", variance_threshold=0.05)
)
param_space.register_dimension(
    DimensionAxis(name="gear_ratio", data_type="continuous", variance_threshold=0.05)
)

primary_holder = {}

def fast_validating_transport(prompt, payload):
    # Binlerce döngüde ağı tıkamamak için deterministik hızlı teşhis
    y_obs = float(payload.get("observed_outcome", 0.0)) if isinstance(payload, dict) else 0.0

    if y_obs > 20.0:
        dim_name = "aerodynamic_drag"
    elif y_obs > 11.0:
        dim_name = "road_incline"
    else:
        dim_name = "latent_feature_1"

    res_payload = {
        "dimension_name": dim_name,
        "statistic": round(y_obs, 2),
        "confidence": 0.95,
        "source_id": "gemini_mock_fast"
    }
    primary_holder["last_res"] = res_payload
    return json.dumps(res_payload)

def fast_peer_validator(query):
    cached = primary_holder.get("last_res")
    parsed = GeminiResearchAgent._parse_llm_output(cached) if cached else None

    y_obs = float(query.observed_outcome) if query and query.observed_outcome else 1.0
    fallback_dim = "aerodynamic_drag" if y_obs > 20.0 else ("road_incline" if y_obs > 11.0 else "latent_feature_1")

    dim_name = fallback_dim
    stat_val = round(y_obs, 2)
    conf_val = 0.92

    if parsed and isinstance(parsed, dict):
        if parsed.get("dimension_name"):
            dim_name = str(parsed["dimension_name"]).strip()
        if parsed.get("statistic") is not None:
            try:
                stat_val = float(parsed["statistic"])
            except (ValueError, TypeError):
                pass

    return {
        "dimension_name": dim_name,
        "statistic": stat_val,
        "confidence": conf_val,
        "source_id": "peer_fast_validator",
    }

research_agent = GeminiResearchAgent(
    api_key="mock_key",
    source_id="gemini_fast",
    transport_fn=fast_validating_transport,
    peer_agents=[fast_peer_validator],
)

engine = (
    EngineBuilder()
    .with_parameter_space(param_space)
    .with_research_agent(research_agent)
    .with_consensus_engine(CrossConsensusEngine(min_independent_sources=2))
    .with_hyperparameters(
        error_threshold=1.8,
        consensus_threshold=0.70,
        regression_tolerance=30.0,
        anchor_sample_size=2,
        anomaly_patience=3,  # Yeni Histerezis Kalkanı
    )
    .build()
)

# 2. 1000 Döngülük Stres Veri Üreteci
TOTAL_CYCLES = 1000
print(f"=== {TOTAL_CYCLES} DÖNGÜLÜK BİLİŞSEL STRES TESTİ BAŞLATILIYOR ===")

loop_stats = {"LOOP_A": 0, "LOOP_C": 0, "LOOP_D": 0}
errors = []
start_time = time.time()

active_dimensions_known = {"engine_rpm", "gear_ratio"}

for step in range(1, TOTAL_CYCLES + 1):
    rpm = random.uniform(1800.0, 3500.0)
    gear = random.choice([1.5, 2.0, 2.5, 3.0])
    noise = random.gauss(0.0, 0.2)  # Sensör gürültüsü

    # Fazlara göre zemin gerçeği formülü
    raw_input = {"engine_rpm": rpm, "gear_ratio": gear}

    if step <= 300:
        # FAZ 1: Standart Düz Yol
        y_true = (rpm / 1000.0) * 2.5 + (gear * 1.2) + noise
    elif step <= 650:
        # FAZ 2: Yokuş Anomalisi (Eksen gizli veya biliniyorsa eklenir)
        incline = 12.0
        y_true = (rpm / 1000.0) * 2.5 + (gear * 1.2) + (incline * 0.45) + noise
        if "road_incline" in engine.parameter_space.dimensions:
            raw_input["road_incline"] = incline
    else:
        # FAZ 3: Ekstrem Rüzgar Anomalisi
        drag = 25.0
        incline = 0.0
        y_true = (rpm / 1000.0) * 2.5 + (gear * 1.2) + (drag * 0.55) + noise
        if "road_incline" in engine.parameter_space.dimensions:
            raw_input["road_incline"] = incline
        if "aerodynamic_drag" in engine.parameter_space.dimensions:
            raw_input["aerodynamic_drag"] = drag

    # Motor adımı
    res = engine.step(raw_input=raw_input, observed_outcome=y_true)

    loop = res.get("loop_executed", "UNKNOWN")
    loop_stats[loop] = loop_stats.get(loop, 0) + 1

    err = res.get("error")
    if isinstance(err, (int, float)):
        errors.append(abs(err))

    # Boyut keşfedildiyse kaydet
    disc = res.get("discovered_dimension")
    if disc:
        active_dimensions_known.add(disc)

    if step % 200 == 0 or step == TOTAL_CYCLES:
        recent_err = sum(errors[-50:]) / max(len(errors[-50:]), 1)
        print(f"Adım {step:4d}/{TOTAL_CYCLES} | Aktif Boyut Sayısı: {len(engine.parameter_space.dimensions)} | Son 50 Adım Ort. Hata: {recent_err:.3f}")

elapsed = time.time() - start_time

# 3. Yıkıcı Unutma (Catastrophic Forgetting) Testi
print("\n" + "="*50)
print("ÇAPRAZ DOĞRULAMA: YIKICI UNUTMA (CATASTROPHIC FORGETTING) KONTROLÜ")
print("="*50)
test_flat_input = {"engine_rpm": 2000.0, "gear_ratio": 2.0}
expected_flat = (2000.0 / 1000.0) * 2.5 + (2.0 * 1.2)  # = 7.40
res_forget = engine.step(raw_input=test_flat_input, observed_outcome=expected_flat)

print(f"Faz 1 Testi (Düz Yol - 1000 adım sonra):")
print(f"  Beklenen: {expected_flat:.2f} | Model Tahmini: {res_forget.get('prediction')} | Hata: {res_forget.get('error')}")
print(f"  Çalışan Döngü: {res_forget.get('loop_executed')} | Durum: {res_forget.get('status')}")

# 4. Genel Rapor
print("\n" + "="*50)
print("STRES TESTİ TELEMETRİ RAPORU")
print("="*50)
print(f"Toplam Süre            : {elapsed:.2f} saniye")
print(f"Throughput             : {TOTAL_CYCLES / elapsed:.1f} adım/saniye")
print(f"Ortalama Mutlak Hata   : {sum(errors) / max(len(errors), 1):.4f}")
print(f"Döngü Dağılımı         :")
for k, v in loop_stats.items():
    pct = (v / TOTAL_CYCLES) * 100
    print(f"  - {k:8s}: {v:4d} kez (%{pct:.1f})")
print(f"Nihai Eksenler         : {list(engine.parameter_space.dimensions.keys())}")