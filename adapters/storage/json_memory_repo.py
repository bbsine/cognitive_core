"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Kalıcı JSON Hafıza Adaptörü.

Bu modül; çekirdek (`core/`) koduna hiçbir müdahalede bulunmadan,
`core.application.ports.IMemoryRepository` SPI sözleşmesini uygulayan,
insan tarafından okunabilir ve dışarıdan elle düzenlenebilir (`storage/knowledge_base.json`)
kalıcı depolama adaptörünü (`JsonMemoryRepository`) tanımlar.

Mimari Özellikler:
1. Dokunulmaz Çekirdek İzolasyonu: `core/` altındaki hiçbir sınıfı değiştirmez.
2. Atomik Disk Yazımı (Atomic Write): Geçici dosya + `os.replace` ile güç kesintisi
   veya çökme anında JSON dosyasının bozulmasını engeller.
3. Dinamik Ağırlık Çıkarımı ve Re-Hydration:
   - Serileştirme sırasında `MicroModule.predict_fn` fonksiyonundan closure analizi
     veya cebirsel birim vektör sondajı (algebraic basis probing) ile `weights` ve
     `bias` katsayılarını otomatik çıkarır.
   - Yükleme (`load_state`) sırasında JSON'daki `weights` ve `bias` değerlerinden
     Y_hat = Σ(w_i * v_i) + b doğrusal tahmin fonksiyonunu dinamik olarak derler.
4. Öz-İyileştirme ve Dayanıklılık (Resilience):
   - Kullanıcı JSON dosyasından bir boyutu (`global_dimensions`) veya ağırlığı elle
     sildiğinde sistemi çökertmez; eksik boyut referansına sahip aktif modülleri
     otomatik olarak H_3'ten ayıklayıp H_2 (Karantina) havuzuna taşır.
"""

from __future__ import annotations
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

from core.application.ports import IMemoryRepository
from core.domain import (
    CognitiveDomainException,
    ComponentVector,
    ConfidenceScore,
    DimensionAxis,
    GlobalParameterSpace,
    MicroModule,
    Observation,
)


class JsonMemoryRepository(IMemoryRepository):
    """
    `IMemoryRepository` SPI portunun `storage/knowledge_base.json` tabanlı,
    dışarıdan düzenlenebilir, atomik yazım ve dinamik fonksiyon derleme (re-hydration)
    destekli kalıcı depolama adaptörü.
    """

    SCHEMA_VERSION: str = "1.0.0"
    DEFAULT_STORAGE_PATH: str = "storage/knowledge_base.json"

    def __init__(
        self,
        file_path: Union[str, Path] = DEFAULT_STORAGE_PATH,
        parameter_space: Optional[GlobalParameterSpace] = None,
        auto_sync: bool = True,
    ) -> None:
        """
        Args:
            file_path: JSON bilgi tabanı dosyasının yolu (Varsayılan: `storage/knowledge_base.json`).
            parameter_space: Çekirdeğin `GlobalParameterSpace` (Ω_global) referansı.
                Verilmezse adaptör kendi iç `GlobalParameterSpace` örneğini yönetir.
            auto_sync: True ise her H_1, H_2, H_3 güncellemesinde durumu otomatik olarak
                diske (`save_state`) yazar.
        """
        self._lock: threading.RLock = threading.RLock()
        self.file_path: Path = Path(file_path)
        self.auto_sync: bool = bool(auto_sync)

        self.parameter_space: GlobalParameterSpace = (
            parameter_space if parameter_space is not None else GlobalParameterSpace()
        )

        # Bellek içi çalışma havuzları
        self._h1_observations: List[Observation] = []
        self._h2_quarantine: Dict[str, MicroModule] = {}
        self._h3_active: Dict[str, MicroModule] = {}

        # Boyutların hangi döngüde keşfedildiğini izleyen üst veri tablosu (Örn: "INIT", "LOOP_C")
        self._dimension_origins: Dict[str, str] = {
            dim_name: "INIT" for dim_name in self.parameter_space.dimensions.keys()
        }

        # Modüllerin bilinen veya çıkarılan (weights, bias) önbelleği
        self._module_linear_params: Dict[str, Tuple[Dict[str, float], float]] = {}

        # Klasör yoksa oluştur; dosya mevcutsa yükle, değilse temiz başlangıç dosyası oluştur.
        self._ensure_storage_directory()
        if self.file_path.exists() and self.file_path.stat().st_size > 0:
            self.load_state(parameter_space=self.parameter_space)
        else:
            self.save_state(parameter_space=self.parameter_space)

    # =========================================================================
    # BAĞLANTI VE KEŞİF ÜST VERİ YÖNETİMİ
    # =========================================================================

    def bind_parameter_space(
        self,
        parameter_space: GlobalParameterSpace,
        default_origin: str = "INIT",
    ) -> None:
        """
        Çekirdeğin `GlobalParameterSpace` (Ω_global) nesnesini adaptöre bağlar ve
        mevcut boyutları kayıt altına alır.
        """
        if not isinstance(parameter_space, GlobalParameterSpace):
            raise CognitiveDomainException(
                "bind_parameter_space yalnızca bir GlobalParameterSpace nesnesi kabul eder."
            )
        with self._lock:
            self.parameter_space = parameter_space
            for dim_name in self.parameter_space.dimensions:
                self._dimension_origins.setdefault(dim_name, default_origin)

    def register_dimension_origin(self, dimension_name: str, discovered_at_loop: str = "LOOP_C") -> None:
        """
        Belirli bir boyutun hangi döngüde (`INIT`, `LOOP_C` vb.) keşfedildiğini işaretler.
        """
        if isinstance(dimension_name, str) and dimension_name.strip():
            with self._lock:
                self._dimension_origins[dimension_name.strip()] = str(discovered_at_loop).strip()

    # =========================================================================
    # 1. HAVUZ (H_1): ÇAPA ÖRNEKLERİ / HAM GÖZLEMLER (IMemoryRepository)
    # =========================================================================

    def save_observation(self, observation: Observation) -> None:
        """
        Yeni gözlemi H_1 havuzuna ekler ve `auto_sync` aktifse JSON dosyasını günceller.
        """
        if not isinstance(observation, Observation):
            raise CognitiveDomainException(
                "JsonMemoryRepository.save_observation yalnızca Observation nesnesi kabul eder.",
                context={"received_type": type(observation).__name__},
            )

        with self._lock:
            self._h1_observations.append(observation)
            if self.auto_sync:
                self.save_state()

    def get_recent_observations(self, limit: int) -> List[Observation]:
        """
        H_1 havuzundaki en güncel `limit` adet gözlemi (en yeniden eskiye) döner.
        """
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise CognitiveDomainException(
                "get_recent_observations 'limit' parametresi bir tam sayı (int) olmalıdır.",
                context={"limit": limit},
            )
        if limit <= 0:
            return []

        with self._lock:
            return list(reversed(self._h1_observations[-limit:]))

    # =========================================================================
    # 2. HAVUZ (H_2): KARANTİNA VE GÖLGE DAL (IMemoryRepository)
    # =========================================================================

    def save_quarantined_module(self, module: MicroModule) -> None:
        """
        Modülü H_2 karantina havuzuna kaydeder, doğrusal ağırlıklarını çıkarır ve
        `auto_sync` aktifse diske yazar.
        """
        if not isinstance(module, MicroModule):
            raise CognitiveDomainException(
                "JsonMemoryRepository.save_quarantined_module yalnızca MicroModule kabul eder.",
                context={"received_type": type(module).__name__},
            )

        with self._lock:
            self._h2_quarantine[module.module_id] = module
            self._module_linear_params[module.module_id] = self._extract_weights_and_bias(module)
            if self.auto_sync:
                self.save_state()

    def get_quarantined_modules(self) -> List[MicroModule]:
        """H_2 karantina havuzundaki tüm modülleri liste olarak döner."""
        with self._lock:
            return list(self._h2_quarantine.values())

    def remove_quarantined_module(self, module_id: str) -> None:
        """Belirtilen modülü H_2 karantina havuzundan güvenli şekilde kaldırır."""
        if not isinstance(module_id, str) or not module_id.strip():
            return

        with self._lock:
            removed = self._h2_quarantine.pop(module_id, None)
            if removed is not None and self.auto_sync:
                self.save_state()

    # =========================================================================
    # 3. HAVUZ (H_3): AKTİF ÇEKİRDEK MODÜLLERİ (IMemoryRepository)
    # =========================================================================

    def save_active_module(self, module: MicroModule) -> None:
        """
        Doğrulanmış uzman modülü H_3 aktif çekirdeğine kaydeder, ağırlıklarını
        serileştirme için önbelleğe alır ve `auto_sync` aktifse diske yazar.
        """
        if not isinstance(module, MicroModule):
            raise CognitiveDomainException(
                "JsonMemoryRepository.save_active_module yalnızca MicroModule kabul eder.",
                context={"received_type": type(module).__name__},
            )

        with self._lock:
            self._h3_active[module.module_id] = module
            self._module_linear_params[module.module_id] = self._extract_weights_and_bias(module)
            if self.auto_sync:
                self.save_state()

    def get_active_modules(self) -> List[MicroModule]:
        """H_3 aktif çekirdek havuzundaki tüm uzman modülleri döner."""
        with self._lock:
            return list(self._h3_active.values())

    def remove_active_module(self, module_id: str) -> None:
        """Belirtilen modülü H_3 aktif çekirdek havuzundan güvenli şekilde kaldırır."""
        if not isinstance(module_id, str) or not module_id.strip():
            return

        with self._lock:
            removed = self._h3_active.pop(module_id, None)
            if removed is not None and self.auto_sync:
                self.save_state()

    # =========================================================================
    # DURUM SERİLEŞTİRME VE GERİ YÜKLEME (STORAGE_SPEC.md BÖLÜM 2 & 3)
    # =========================================================================

    def save_state(
        self,
        parameter_space: Optional[GlobalParameterSpace] = None,
        active_modules: Optional[List[MicroModule]] = None,
        quarantine_modules: Optional[List[MicroModule]] = None,
        anchor_samples: Optional[List[Observation]] = None,
    ) -> Path:
        """
        Çekirdeğin güncel boyutlarını (Ω_global), H_3 aktif modüllerini, H_2 karantina
        havuzunu ve H_1 çapa örneklerini `STORAGE_SPEC.md` şemasına uygun olarak
        atomik yazım (atomic write) ile `knowledge_base.json` dosyasına kaydeder.

        Returns:
            Yazılan JSON dosyasının `Path` nesnesi.
        """
        with self._lock:
            if parameter_space is not None:
                self.parameter_space = parameter_space

            if active_modules is not None:
                self._h3_active = {m.module_id: m for m in active_modules}
                for m in active_modules:
                    self._module_linear_params[m.module_id] = self._extract_weights_and_bias(m)

            if quarantine_modules is not None:
                self._h2_quarantine = {m.module_id: m for m in quarantine_modules}
                for m in quarantine_modules:
                    self._module_linear_params[m.module_id] = self._extract_weights_and_bias(m)

            if anchor_samples is not None:
                self._h1_observations = list(anchor_samples)

            # Ω_global içindeki yeni boyutların keşif kaynağını güncelle
            for dim_name in self.parameter_space.dimensions.keys():
                if dim_name not in self._dimension_origins:
                    self._dimension_origins[dim_name] = "LOOP_C"

            payload: Dict[str, Any] = {
                "version": self.SCHEMA_VERSION,
                "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "global_dimensions": [
                    {
                        "name": axis.name,
                        "data_type": axis.data_type,
                        "discovered_at_loop": self._dimension_origins.get(axis.name, "LOOP_C"),
                    }
                    for axis in self.parameter_space.dimensions.values()
                ],
                "active_modules_h3": [
                    self._serialize_module(mod, status="ACTIVE")
                    for mod in self._h3_active.values()
                ],
                "quarantine_modules_h2": [
                    self._serialize_module(mod, status="QUARANTINED")
                    for mod in self._h2_quarantine.values()
                ],
                "anchor_samples_h1": [
                    {
                        "input_vector": dict(obs.features.values),
                        "observed_outcome": obs.observed_outcome,
                    }
                    for obs in self._h1_observations
                ],
            }

            self._atomic_write_json(payload)
            return self.file_path

    def load_state(
        self,
        parameter_space: Optional[GlobalParameterSpace] = None,
    ) -> Dict[str, Any]:
        """
        `knowledge_base.json` dosyasını okur, ağırlıkları dinamik `predict_fn`
        fonksiyonlarına derler (Re-Hydration) ve çekirdeğin Domain nesnelerine dönüştürür.

        Dayanıklılık Kuralları:
        1. Dosya bulunamazsa veya içi boşsa hata fırlatmaz; boş bellek havuzlarıyla
           temiz bir başlangıç durumu döner.
        2. Kullanıcı JSON dosyasından bir boyutu (`global_dimensions`) veya kuralı
           elle silmişse sistemi çökertmez; silinen boyuta referans veren H_3
           modüllerini otomatik olarak ayıklayıp H_2 (Karantina) havuzuna taşır.

        Args:
            parameter_space: Güncellenecek `GlobalParameterSpace` nesnesi (opsiyonel).

        Returns:
            Yüklenen domain nesnelerini ve varsa karantinaya alınan kırık referanslı
            modül kimliklerini içeren durum sözlüğü.
        """
        with self._lock:
            if parameter_space is not None:
                self.parameter_space = parameter_space

            if not self.file_path.exists():
                self._reset_memory_pools()
                return self._build_state_summary(quarantined_due_to_missing_dims=[])

            raw_content = self.file_path.read_text(encoding="utf-8").strip()
            if not raw_content:
                self._reset_memory_pools()
                return self._build_state_summary(quarantined_due_to_missing_dims=[])

            try:
                data: Dict[str, Any] = json.loads(raw_content)
            except json.JSONDecodeError:
                # Bozuk JSON durumunda çökme yerine güvenli boş havuzlarla başla
                self._reset_memory_pools()
                return self._build_state_summary(quarantined_due_to_missing_dims=[])

            if not isinstance(data, dict):
                self._reset_memory_pools()
                return self._build_state_summary(quarantined_due_to_missing_dims=[])

            # 1. Global Boyutları (Ω_global) Yükle
            loaded_dimensions: Dict[str, DimensionAxis] = {}
            loaded_origins: Dict[str, str] = {}
            raw_dimensions = data.get("global_dimensions", [])

            if isinstance(raw_dimensions, list):
                for item in raw_dimensions:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    data_type = item.get("data_type", "continuous")
                    discovered_loop = item.get("discovered_at_loop", "INIT")
                    variance_threshold = item.get("variance_threshold", 0.0)

                    if not isinstance(name, str) or not name.strip():
                        continue
                    try:
                        axis = DimensionAxis(
                            name=name.strip(),
                            data_type=str(data_type).strip() or "continuous",
                            variance_threshold=float(variance_threshold),
                        )
                        loaded_dimensions[axis.name] = axis
                        loaded_origins[axis.name] = str(discovered_loop).strip() or "INIT"
                    except CognitiveDomainException:
                        continue

            self.parameter_space.dimensions.clear()
            for axis in loaded_dimensions.values():
                self.parameter_space.register_dimension(axis)
            self._dimension_origins = loaded_origins
            known_dimension_names: Set[str] = set(loaded_dimensions.keys())

            # 2. H_2 Karantina Modüllerini Yükle
            loaded_h2: Dict[str, MicroModule] = {}
            loaded_params: Dict[str, Tuple[Dict[str, float], float]] = {}

            raw_h2_list = data.get("quarantine_modules_h2", [])
            if isinstance(raw_h2_list, list):
                for raw_mod in raw_h2_list:
                    hydrated = self._rehydrate_module(raw_mod)
                    if hydrated is not None:
                        mod, weights, bias, _ = hydrated
                        loaded_h2[mod.module_id] = mod
                        loaded_params[mod.module_id] = (weights, bias)

            # 3. H_3 Aktif Modüllerini Yükle & Eksik Boyut Referanslarını Denetle
            loaded_h3: Dict[str, MicroModule] = {}
            demoted_ids: List[str] = []

            raw_h3_list = data.get("active_modules_h3", [])
            if isinstance(raw_h3_list, list):
                for raw_mod in raw_h3_list:
                    hydrated = self._rehydrate_module(raw_mod)
                    if hydrated is None:
                        continue

                    mod, weights, bias, had_integrity_warning = hydrated
                    loaded_params[mod.module_id] = (weights, bias)

                    # Dayanıklılık Denetimi:
                    # Modülün active_dimensions veya weights anahtarlarından herhangi biri
                    # global_dimensions listesinden elle silinmişse veya tutarsızsa
                    # modülü H_3 yerine H_2 karantinasına taşı!
                    referenced_dims = set(mod.active_dimensions) | set(weights.keys())
                    missing_in_global = referenced_dims - known_dimension_names
                    missing_weights = set(mod.active_dimensions) - set(weights.keys())

                    if missing_in_global or missing_weights or had_integrity_warning:
                        loaded_h2[mod.module_id] = mod
                        demoted_ids.append(mod.module_id)
                    else:
                        loaded_h3[mod.module_id] = mod

            # 4. H_1 Çapa Örneklerini (Anchor Samples) Yükle
            loaded_h1: List[Observation] = []
            raw_h1_list = data.get("anchor_samples_h1", [])
            if isinstance(raw_h1_list, list):
                for idx, raw_obs in enumerate(raw_h1_list):
                    if not isinstance(raw_obs, dict):
                        continue
                    raw_vec = raw_obs.get("input_vector")
                    if not isinstance(raw_vec, dict) or len(raw_vec) == 0:
                        continue

                    # Eğer kullanıcı global_dimensions'dan bir boyutu sildiyse ve
                    # çapa verisinde sadece geçerli sayısal değerler varsa güvenli şekilde filtrele
                    clean_vec_dict: Dict[str, float] = {}
                    for k, v in raw_vec.items():
                        if (
                            isinstance(k, str)
                            and k.strip()
                            and isinstance(v, (int, float))
                            and not isinstance(v, bool)
                        ):
                            fv = float(v)
                            if not (math.isnan(fv) or math.isinf(fv)):
                                clean_vec_dict[k.strip()] = fv

                    if not clean_vec_dict:
                        continue

                    try:
                        obs = Observation(
                            id=str(raw_obs.get("id", f"anchor_h1_{idx + 1}_{uuid.uuid4().hex[:6]}")),
                            features=ComponentVector(values=clean_vec_dict),
                            observed_outcome=raw_obs.get("observed_outcome"),
                            timestamp=float(raw_obs.get("timestamp", time.time())),
                        )
                        loaded_h1.append(obs)
                    except CognitiveDomainException:
                        continue

            self._h1_observations = loaded_h1
            self._h2_quarantine = loaded_h2
            self._h3_active = loaded_h3
            self._module_linear_params = loaded_params

            return self._build_state_summary(quarantined_due_to_missing_dims=demoted_ids)

    # =========================================================================
    # AĞIRLIK ÇIKARIMI VE DİNAMİK FONKSİYON DERLEME (RE-HYDRATION)
    # =========================================================================

    @staticmethod
    def compile_linear_predictor(
        weights: Dict[str, float],
        bias: float,
    ) -> Callable[[ComponentVector], float]:
        """
        JSON'dan okunan `weights` ve `bias` değerlerini kullanarak
        Y_hat = Σ(w_i * v_i) + b formülünü işleten dinamik bir `predict_fn`
        fonksiyonu derler.
        """
        frozen_weights: Dict[str, float] = {
            str(k): float(v) for k, v in weights.items()
        }
        frozen_bias: float = float(bias)

        def dynamic_predict_fn(vector: ComponentVector) -> float:
            prediction: float = frozen_bias
            for dim_name, weight in frozen_weights.items():
                prediction += weight * vector.values.get(dim_name, 0.0)
            return prediction

        # Serileştirme sırasında hızlı okuma için fonksiyon nesnesine üst veri ekle
        setattr(dynamic_predict_fn, "weights", dict(frozen_weights))
        setattr(dynamic_predict_fn, "bias", frozen_bias)
        return dynamic_predict_fn

    def _extract_weights_and_bias(
        self,
        module: MicroModule,
    ) -> Tuple[Dict[str, float], float]:
        """
        Çekirdek (`core/`) koduna dokunmadan bir `MicroModule` nesnesinin
        doğrusal ağırlıklarını (`weights`) ve sapmasını (`bias`) 3 aşamalı
        stratejiyle çıkarır:

        1. Fonksiyon Üst Verisi: `predict_fn.weights` ve `predict_fn.bias` öznitelikleri.
        2. Closure Hücre Analizi: `PurePythonNeuralEngine` tarafından üretilen
           `frozen_weights` ve `frozen_bias` serbest değişkenlerinin (`__closure__`) okunması.
        3. Cebirsel Birim Vektör Sondajı (Algebraic Basis Probing):
           Herhangi bir doğrusal lambda fonksiyonuna sıfır vektörü V_0 ve birim
           baz vektörleri V_{e_i} göndererek b = f(0) ve w_i = f(e_i) - b değerlerinin
           matematiksel olarak çözülmesi.
        """
        fn = module.predict_fn
        sorted_dims = sorted(module.active_dimensions)

        # Aşama 1: Açık fonksiyon öznitelikleri (Re-hydrated modüller)
        attr_weights = getattr(fn, "weights", None)
        attr_bias = getattr(fn, "bias", None)
        if isinstance(attr_weights, dict) and isinstance(attr_bias, (int, float)):
            clean_w = {
                dim: float(attr_weights.get(dim, 0.0)) for dim in sorted_dims
            }
            return clean_w, float(attr_bias)

        # Aşama 2: Closure hücre analizi (PurePythonNeuralEngine uyumluluğu)
        if getattr(fn, "__closure__", None) and getattr(fn, "__code__", None):
            freevars = fn.__code__.co_freevars
            closure_Map = {
                var_name: cell.cell_contents
                for var_name, cell in zip(freevars, fn.__closure__)
            }
            if "frozen_weights" in closure_Map and "frozen_bias" in closure_Map:
                cw = closure_Map["frozen_weights"]
                cb = closure_Map["frozen_bias"]
                if isinstance(cw, dict) and isinstance(cb, (int, float)):
                    clean_w = {dim: float(cw.get(dim, 0.0)) for dim in sorted_dims}
                    return clean_w, float(cb)

        # Aşama 3: Cebirsel Birim Vektör Sondajı (Black-Box Linear Probing)
        try:
            zero_dict = {dim: 0.0 for dim in sorted_dims}
            zero_vec = ComponentVector(values=zero_dict)
            bias_val = fn(zero_vec)

            if isinstance(bias_val, (int, float)) and not isinstance(bias_val, bool):
                b_float = float(bias_val)
                if not (math.isnan(b_float) or math.isinf(b_float)):
                    probed_weights: Dict[str, float] = {}
                    for target_dim in sorted_dims:
                        basis_dict = {
                            dim: (1.0 if dim == target_dim else 0.0)
                            for dim in sorted_dims
                        }
                        basis_vec = ComponentVector(values=basis_dict)
                        out_val = fn(basis_vec)
                        if isinstance(out_val, (int, float)) and not isinstance(out_val, bool):
                            w_i = float(out_val) - b_float
                            probed_weights[target_dim] = (
                                0.0 if (math.isnan(w_i) or math.isinf(w_i)) else round(w_i, 12)
                            )
                        else:
                            probed_weights[target_dim] = 0.0
                    return probed_weights, round(b_float, 12)
        except Exception:
            pass

        # Geri dönüş (Fallback)
        if module.module_id in self._module_linear_params:
            return self._module_linear_params[module.module_id]

        return ({dim: 0.0 for dim in sorted_dims}, 0.0)

    def _rehydrate_module(
        self,
        raw_mod: Any,
    ) -> Optional[Tuple[MicroModule, Dict[str, float], float, bool]]:
        """
        JSON'daki tekil bir modül sözlüğünü `MicroModule` nesnesine dönüştürür.
        Eksik alan veya sınır ihlali varsa `had_integrity_warning = True` döner.
        """
        if not isinstance(raw_mod, dict):
            return None

        module_id = raw_mod.get("module_id")
        if not isinstance(module_id, str) or not module_id.strip():
            return None
        module_id = module_id.strip()

        raw_active_dims = raw_mod.get("active_dimensions")
        raw_weights = raw_mod.get("weights")
        had_integrity_warning: bool = False

        # Ağırlıkları temizle
        clean_weights: Dict[str, float] = {}
        if isinstance(raw_weights, dict):
            for k, v in raw_weights.items():
                if (
                    isinstance(k, str)
                    and k.strip()
                    and isinstance(v, (int, float))
                    and not isinstance(v, bool)
                ):
                    fv = float(v)
                    if not (math.isnan(fv) or math.isinf(fv)):
                        clean_weights[k.strip()] = fv
                    else:
                        had_integrity_warning = True
                else:
                    had_integrity_warning = True
        else:
            had_integrity_warning = True

        # Aktif boyutları temizle
        active_dims: Set[str] = set()
        if isinstance(raw_active_dims, list):
            for d in raw_active_dims:
                if isinstance(d, str) and d.strip():
                    active_dims.add(d.strip())
                else:
                    had_integrity_warning = True
        else:
            had_integrity_warning = True

        # Kullanıcı active_dimensions'ı sildiyse ancak weights duruyorsa boyutları weights'ten kurtar
        if not active_dims and clean_weights:
            active_dims = set(clean_weights.keys())
            had_integrity_warning = True

        if not active_dims:
            return None

        # Bias değerini doğrula
        raw_bias = raw_mod.get("bias", 0.0)
        if (
            isinstance(raw_bias, (int, float))
            and not isinstance(raw_bias, bool)
            and not math.isnan(float(raw_bias))
            and not math.isinf(float(raw_bias))
        ):
            bias_val = float(raw_bias)
        else:
            bias_val = 0.0
            had_integrity_warning = True

        # ConfidenceScore doğrulaması (Invariant #1: 0.0 <= P < 0.9999)
        raw_conf = raw_mod.get("confidence_score", 0.50)
        if (
            isinstance(raw_conf, (int, float))
            and not isinstance(raw_conf, bool)
            and not math.isnan(float(raw_conf))
            and not math.isinf(float(raw_conf))
        ):
            score_val = float(raw_conf)
        else:
            score_val = 0.50
            had_integrity_warning = True

        max_valid_score = math.nextafter(ConfidenceScore.DEFAULT_P_MAX, 0.0)
        if not (ConfidenceScore.DEFAULT_P_MIN <= score_val < ConfidenceScore.DEFAULT_P_MAX):
            # Kullanıcı elle P >= 1.0 veya negatif yazdıysa çökmemek için sınır içine çek ve uyar
            score_val = max(ConfidenceScore.DEFAULT_P_MIN, min(score_val, max_valid_score))
            had_integrity_warning = True

        confidence_vo = ConfidenceScore(score=score_val)
        predict_fn = self.compile_linear_predictor(weights=clean_weights, bias=bias_val)

        module = MicroModule(
            module_id=module_id,
            active_dimensions=active_dims,
            confidence=confidence_vo,
            predict_fn=predict_fn,
        )
        return module, clean_weights, bias_val, had_integrity_warning

    # =========================================================================
    # DAHİLİ SERİLEŞTİRME VE ATOMİK DOSYA YARDIMCILARI
    # =========================================================================

    def _serialize_module(self, module: MicroModule, status: str) -> Dict[str, Any]:
        """Tek bir MicroModule nesnesini `STORAGE_SPEC.md` şemasına dönüştürür."""
        weights, bias = self._extract_weights_and_bias(module)
        self._module_linear_params[module.module_id] = (weights, bias)

        return {
            "module_id": module.module_id,
            "active_dimensions": sorted(module.active_dimensions),
            "weights": {dim: weights.get(dim, 0.0) for dim in sorted(module.active_dimensions)},
            "bias": bias,
            "confidence_score": module.confidence.score,
            "status": status,
        }

    def _ensure_storage_directory(self) -> None:
        """Depolama klasörü (`storage/`) mevcut değilse çalışma anında oluşturur."""
        if self.file_path.parent:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _atomic_write_json(self, payload: Dict[str, Any]) -> None:
        """
        JSON içeriğini önce aynı dizindeki geçici bir dosyaya yazar, `fsync` ile
        diske boşaltır ve `os.replace` ile hedef dosyayı atomik olarak günceller.
        """
        self._ensure_storage_directory()
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)

        fd, tmp_path_str = tempfile.mkstemp(
            dir=str(self.file_path.parent),
            prefix=f".{self.file_path.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(serialized)
                tmp_file.write("\n")
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(str(tmp_path), str(self.file_path))
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    def _reset_memory_pools(self) -> None:
        """Bellek içi tüm havuzları temiz bir başlangıç durumuna sıfırlar."""
        self._h1_observations.clear()
        self._h2_quarantine.clear()
        self._h3_active.clear()
        self._module_linear_params.clear()

    def _build_state_summary(
        self,
        quarantined_due_to_missing_dims: List[str],
    ) -> Dict[str, Any]:
        """`load_state` dönüş değerini yapılandırır."""
        return {
            "parameter_space": self.parameter_space,
            "global_dimensions": list(self.parameter_space.dimensions.values()),
            "active_modules_h3": list(self._h3_active.values()),
            "quarantine_modules_h2": list(self._h2_quarantine.values()),
            "anchor_samples_h1": list(self._h1_observations),
            "quarantined_due_to_missing_dimensions": list(quarantined_due_to_missing_dims),
        }


__all__ = ["JsonMemoryRepository"]