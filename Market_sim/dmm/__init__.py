"""
Deep Markov Model (DMM) Module

A neural network-based extension of Hidden Markov Models for learning
context-dependent regime dynamics in tokenized vs traditional CRE markets.

Main Components:
    - DeepMarkovModel: Core ML model with training/inference
    - TransitionNetwork: Learns regime transitions
    - EmissionNetwork: Generates price/volatility distributions
    - InferenceNetwork: Infers regimes from observations

Quick Start:
    >>> from dmm import DeepMarkovModel
    >>> 
    >>> # Train model
    >>> # python3 dmm/training/train_dmm_with_qfclient.py
    >>> 
    >>> # Load and use
    >>> dmm = DeepMarkovModel()
    >>> dmm.load('outputs/deep_markov_model_qfclient.pt')
    >>> 
    >>> # Predict next regime
    >>> next_regime, probs = dmm.predict_next_regime(
    ...     current_regime='calm',
    ...     context={'is_tokenized': 1.0, 'time_normalized': 0.5, 'adoption_rate': 0.7}
    ... )

Documentation:
    - README.md: Main documentation
    - docs/: Additional guides and documentation

Folder Structure:
    - core/: Core model implementation
    - training/: Training scripts
    - utils/: Utility functions and data loaders
    - docs/: Documentation files

See README.md for detailed usage and examples.
"""

from dmm.core.deep_markov_model import (
    DeepMarkovModel,
    TransitionNetwork,
    EmissionNetwork,
    InferenceNetwork,
    TORCH_AVAILABLE
)

__version__ = '1.0.0'
__author__ = 'Market Simulation Team'

__all__ = [
    'DeepMarkovModel',
    'TransitionNetwork',
    'EmissionNetwork',
    'InferenceNetwork',
    'TORCH_AVAILABLE'
]
