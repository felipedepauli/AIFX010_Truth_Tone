# src/hoaxhertz/features/interface.py
# -*- coding: utf-8 -*-
"""
Generic feature extraction interface for the pipeline.

- Strongly-typed inputs and outputs using Pydantic
- A simple ABC that any extractor (ambient, ENF, MFCC, ...) can implement
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class PreprocInput(BaseModel):
    """
    Standardized preprocessed bundle shared with feature extractors.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    y: np.ndarray = Field(..., description="Mono waveform, float, range approx. [-1, 1].")
    sr: int = Field(..., gt=0, description="Sample rate in Hz.")
    frames: Optional[np.ndarray] = Field(
        default=None,
        description="Framed signal with shape (frame_len, n_frames) if available."
    )
    n_fft: Optional[int] = Field(default=None, description="Frame length in samples.")
    hop: Optional[int] = Field(default=None, description="Hop length in samples.")


class FeatureOutput(BaseModel):
    """
    Generic feature output compatible with downstream ML and detectors.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector: Optional[np.ndarray] = Field(
        default=None,
        description="Fixed-length feature vector for clip-level ML."
    )
    framewise: Optional[Dict[str, np.ndarray]] = Field(
        default=None,
        description="Frame-level matrices (e.g., log-mel T×M, times T, etc.)."
    )
    meta: Dict[str, Any] = Field(default_factory=dict, description="Auxiliary metadata.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Extraction parameters.")


class FeatureConfigBase(BaseModel):
    """
    Base configuration model for feature extractors.
    Extend this in each extractor with specific parameters.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "feature"
    return_framewise: bool = False


class FeatureExtractor(ABC):
    """
    Abstract interface for all feature extractors.

    Typical usage:
        extractor = ConcreteExtractor()
        extractor.configure(cfg_model)
        out = extractor.extract(pre)
    """

    def __init__(self) -> None:
        self._cfg: Optional[FeatureConfigBase] = None

    @abstractmethod
    def configure(self, cfg: FeatureConfigBase) -> None:
        """
        Store and validate the configuration for this extractor.
        """
        ...

    @abstractmethod
    def extract(self, pre: PreprocInput) -> FeatureOutput:
        """
        Compute feature(s) from preprocessed inputs.
        """
        ...

    @property
    def cfg(self) -> FeatureConfigBase:
        if self._cfg is None:
            raise RuntimeError("Extractor is not configured. Call configure(cfg) first.")
        return self._cfg
