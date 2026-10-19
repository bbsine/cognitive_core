"""
Bilişsel Çekirdek Motoru (Cognitive Core Engine) - Application Katmanı Paketi.

Bu paket; dış adaptörler için Soyut Servis Sağlayıcı Portlarını (SPI) ve
bilişsel döngüleri (Döngü A, B, C, D) yöneten CognitiveOrchestrator
uygulama servisini dış katmanlara açar.
"""

from core.application.ports import (
    IMemoryRepository,
    INeuralEnginePort,
    IResearchAgent,
)
from core.application.services import CognitiveOrchestrator

__all__ = [
    # SPI Ports
    "IMemoryRepository",
    "IResearchAgent",
    "INeuralEnginePort",
    # Application Services / Use Case Orchestrators
    "CognitiveOrchestrator",
]