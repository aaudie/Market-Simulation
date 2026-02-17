#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training with detailed diagnostics to understand what's happening
"""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
dmm_dir = script_dir.parent
parent_dir = dmm_dir.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import torch
from dmm import DeepMarkovModel
from dmm.utils.qfclient_data_loader import (
    load_reit_data,
    prepare_dmm_training_data,
    generate_synthetic_reit_data,
    QFCLIENT_AVAILABLE
)
from sim.data_loader import load_cre_csv


def load_data():
    """Load training data"""
    print("Loading data...")
    
    # Traditional CRE
    csv_path = parent_dir / "data" / "cre_monthly.csv"
    if csv_path.exists():
        from sim.types import HistoricalPoint
        history = load_cre_csv(str(csv_path))
        trad_prices = np.array([p.price for p in history])
    else:
        trad_prices = generate_synthetic_reit_data(864)
    
    # Tokenized REIT
    if QFCLIENT_AVAILABLE:
        try:
            token_prices = load_reit_data("VNQ", years=20, interval="monthly")
        except:
            token_prices = generate_synthetic_reit_data(252)
    else:
        token_prices = generate_synthetic_reit_data(252)
    
    # Prepare training data
    training_data = prepare_dmm_training_data(
        traditional_data=trad_prices,
        tokenized_data=token_prices,
        window_size=72,
        stride=12
    )
    
    return training_data


def train_with_diagnostics():
    """Train model with detailed diagnostics"""
    
    data = load_data()
    
    print("\n" + "="*70)
    print("TRAINING WITH DIAGNOSTICS")
    print("="*70)
    
    # Initialize model
    model = DeepMarkovModel(
        regime_names=['calm', 'neutral', 'volatile', 'panic'],
        context_dim=3,
        hidden_dim=128,
        learning_rate=1e-3
    )
    
    print(f"\nModel Configuration:")
    print(f"  Device: {model.device}")
    print(f"  Learning rate: 1e-3")
    print(f"  Warmup: first 20% of epochs (beta=0)")
    print(f"  Entropy target: 0.7")
    print(f"  Non-uniform initial prior")
    
    # Prepare data
    prepared_data = model.prepare_data(
        data['prices'],
        data['is_tokenized'],
        data['adoption_rates']
    )
    
    observations = prepared_data['observations']
    context = prepared_data['context']
    
    print(f"\nData shapes:")
    print(f"  Observations: {observations.shape}")
    print(f"  Context: {context.shape}")
    
    # Check data statistics after normalization
    print(f"\nNormalized data statistics:")
    print(f"  Returns - mean: {observations[:,:,0].mean():.6f}, std: {observations[:,:,0].std():.6f}")
    print(f"  Volatility - mean: {observations[:,:,1].mean():.6f}, std: {observations[:,:,1].std():.6f}")
    
    # Training with detailed logging
    epochs = 200
    batch_size = 16
    n_sequences = observations.shape[0]
    
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}")
    print("="*70)
    
    for epoch in range(epochs):
        # Compute beta and tau
        warmup_epochs = int(epochs * 0.2)
        if epoch < warmup_epochs:
            beta = 0.0
        else:
            progress = (epoch - warmup_epochs) / (epochs * 0.7)
            beta = min(1.0, progress)
        
        tau = max(0.5, 1.0 - (epoch / epochs) * 0.5)
        
        # Training
        model.transition_net.train()
        model.emission_net.train()
        model.inference_net.train()
        
        epoch_metrics = {
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'entropy': 0.0,
            'diversity_loss': 0.0,
            'total_loss': 0.0
        }
        n_batches = 0
        
        # Track gradients
        grad_norms = []
        
        indices = np.random.permutation(n_sequences)
        for i in range(0, n_sequences, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_obs = observations[batch_idx]
            batch_ctx = context[batch_idx]
            
            model.optimizer.zero_grad()
            loss, metrics = model.compute_elbo(batch_obs, batch_ctx, beta, tau)
            loss.backward()
            
            # Track gradient norm
            total_norm = 0
            for p in model.inference_net.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(total_norm ** 0.5)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(model.transition_net.parameters()) +
                list(model.emission_net.parameters()) +
                list(model.inference_net.parameters()),
                max_norm=5.0
            )
            
            model.optimizer.step()
            
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            n_batches += 1
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
        
        avg_grad_norm = np.mean(grad_norms)
        
        # Store history
        model.training_history['loss'].append(epoch_metrics['total_loss'])
        model.training_history['reconstruction_loss'].append(epoch_metrics['reconstruction_loss'])
        model.training_history['kl_loss'].append(epoch_metrics['kl_loss'])
        model.training_history['entropy'].append(epoch_metrics['entropy'])
        model.training_history['diversity_loss'].append(epoch_metrics['diversity_loss'])
        
        # Detailed logging every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            # Check regime usage
            with torch.no_grad():
                regime_probs = model.inference_net(observations, context)
                regime_usage = regime_probs.mean(dim=[0, 1]).cpu().numpy()
            
            print(f"\nEpoch {epoch:3d}/{epochs}")
            print(f"  Loss: {epoch_metrics['total_loss']:.4f} | "
                  f"Recon: {epoch_metrics['reconstruction_loss']:.4f} | "
                  f"KL: {epoch_metrics['kl_loss']:.4f}")
            print(f"  Ent: {epoch_metrics['entropy']:.4f} | "
                  f"Div: {epoch_metrics['diversity_loss']:.4f} | "
                  f"GradNorm: {avg_grad_norm:.4f}")
            print(f"  Beta: {beta:.3f} | Tau: {tau:.3f}")
            print(f"  Regime usage: Calm={regime_usage[0]:.3f}, "
                  f"Neutral={regime_usage[1]:.3f}, "
                  f"Volatile={regime_usage[2]:.3f}, "
                  f"Panic={regime_usage[3]:.3f}")
    
    # Save model
    save_path = str(parent_dir / "outputs" / "deep_markov_model_diagnostic.pt")
    model.save(save_path)
    print(f"\n\nModel saved: {save_path}")
    print("="*70)
    
    return model


if __name__ == "__main__":
    model = train_with_diagnostics()
