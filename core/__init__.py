"""
LOB Intensity Simulator Core Package
Implements Toke & Yoshida (2016) limit order book intensity models.
"""

from .intensity_models import (
    IntensityModel,
    MarketOrderIntensityModel, 
    LimitOrderIntensityModel
)

from .placement_model import LimitOrderPlacementModel

from .simulator import OrderFlowSimulator

from .data_handler import DataHandler

__version__ = "1.0.0"
__author__ = "LOB Intensity Simulator Team"
__description__ = "Limit Order Book Intensity Models - Toke & Yoshida (2016)"

__all__ = [
    'IntensityModel',
    'MarketOrderIntensityModel',
    'LimitOrderIntensityModel', 
    'LimitOrderPlacementModel',
    'OrderFlowSimulator',
    'DataHandler'
]
