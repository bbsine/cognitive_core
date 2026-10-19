"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Domain İstisna Hiyerarşisi.

Bu modül, çekirdeğin epistemik kuralları, regresyon sınırları veya
parametre uzayı bütünlüğü ihlal edildiğinde fırlatılan saf domain
hatalarını tanımlar. Sıfır harici kütüphane bağımlılığı ilkesine tabidir.
"""

from __future__ import annotations
from typing import Any, Dict, Optional


class CognitiveDomainException(Exception):
    """ 
    Bilişsel Çekirdek Domain katmanındaki tüm kural ve invaryant
    ihlalleri için temel (base) istisna sınıfı.
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.context: Dict[str, Any] = dict(context) if context is not None else {}

    def __str__(self) -> str:
        if not self.context:
            return self.message
        context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} [Bağlam: {context_str}]"


class EpistemicInvariantViolationException(CognitiveDomainException):
    """
    Epistemik yanılabilirlik yasaları (örn. P >= 1.0 yasağı, geçersiz
    asimptotik öğrenme oranı η veya ceza çarpanı γ) çiğnendiğinde fırlatılır.

    Referans: MEMORY.md - Invariant #1 (P < 1.0 Kuralı)
    """


class RegressionFailureException(CognitiveDomainException):
    """
    H_2 (Gölge Dal / Karantina) havuzunda sentezlenen aday modülün (m_yeni),
    geçmişte doğrulanmış çapa verileri (anchor data) üzerinde yapılan geriye
    dönük stabilite testini (Loss_eski <= Kabul_Toleransı) geçemediğinde fırlatılır.

    Referans: MEMORY.md - Invariant #2 (Geriye Dönük Regresyon Koruması)
    """


class DimensionNotFoundException(CognitiveDomainException):
    """
    Global parametre uzayında (Ω_global) veya bileşen vektöründe (V)
    aranan boyut ekseni (θ / v_i) bulunamadığında fırlatılır.
    """