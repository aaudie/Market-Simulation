"""
Deep Markov Model for Market Regime Dynamics

Extends the discrete HMM with neural networks to learn:
1. Transition dynamics between regimes
2. Emission distributions (price/volatility given regime)
3. Context-dependent regime behavior (traditional vs tokenized)

Architecture:
- Transition Network: Learns P(regime_t+1 | regime_t, context)
- Emission Network: Learns P(returns, volatility | regime_t, context)
- Inference Network: Estimates posterior P(regime_t | observations)

Training Strategy (Two-Phase):
- Phase 1 (Supervised): Train inference + emission networks on heuristic regime labels
- Phase 2 (VAE): Fine-tune all networks with variational objective
"""

import math
from typing import List, Tuple, Dict, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")


# =============================================================================
# Neural Network Components
# =============================================================================

class TransitionNetwork(nn.Module):
    """
    Neural network for learning regime transitions.
    
    Outputs transition probabilities P(regime_t+1 | regime_t, context)
    Context includes: market type (tokenized flag), time index, adoption rate
    """
    
    def __init__(self, n_regimes: int = 4, context_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.n_regimes = n_regimes
        
        # Input: one-hot current regime + context features
        input_dim = n_regimes + context_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, regime_idx: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            regime_idx: Current regime index [batch_size]
            context: Context features [batch_size, context_dim]
        
        Returns:
            Transition probabilities [batch_size, n_regimes]
        """
        # One-hot encode regime
        regime_onehot = F.one_hot(regime_idx, num_classes=self.n_regimes).float()
        
        # Concatenate regime and context
        x = torch.cat([regime_onehot, context], dim=-1)
        
        return self.net(x)


class EmissionNetwork(nn.Module):
    """
    Neural network for learning emission distributions.
    
    Outputs parameters for distribution P(observation | regime, context)
    Observation = (log_return, realized_volatility)
    """
    
    def __init__(self, n_regimes: int = 4, context_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.n_regimes = n_regimes
        
        input_dim = n_regimes + context_dim
        
        # Network outputs: mean_return, log_std_return, mean_vol, log_std_vol
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # [mu_r, log_sigma_r, mu_v, log_sigma_v]
        )
    
    def forward(self, regime_idx: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            regime_idx: Current regime [batch_size]
            context: Context features [batch_size, context_dim]
        
        Returns:
            Dict with 'return_dist' and 'vol_dist' (Normal distributions)
        """
        regime_onehot = F.one_hot(regime_idx, num_classes=self.n_regimes).float()
        x = torch.cat([regime_onehot, context], dim=-1)
        
        params = self.net(x)
        
        # Split parameters
        mu_return = params[:, 0]
        log_std_return = params[:, 1]
        mu_vol = params[:, 2]
        log_std_vol = params[:, 3]
        
        # Create distributions (use exp for std to ensure positivity)
        return_dist = Normal(mu_return, torch.exp(log_std_return))
        vol_dist = Normal(mu_vol, torch.exp(log_std_vol))
        
        return {
            'return_dist': return_dist,
            'vol_dist': vol_dist,
            'mu_return': mu_return,
            'sigma_return': torch.exp(log_std_return),
            'mu_vol': mu_vol,
            'sigma_vol': torch.exp(log_std_vol)
        }


class InferenceNetwork(nn.Module):
    """
    Inference network for posterior estimation.
    
    Given observation sequence, estimates P(regime_t | observations)
    Uses bidirectional LSTM to incorporate past and future context
    """
    
    def __init__(self, n_regimes: int = 4, obs_dim: int = 2, context_dim: int = 3, 
                 hidden_dim: int = 64):
        super().__init__()
        self.n_regimes = n_regimes
        
        # BiLSTM to process observation sequence
        self.lstm = nn.LSTM(
            input_size=obs_dim + context_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layer: LSTM hidden -> regime probabilities
        self.output = nn.Linear(hidden_dim * 2, n_regimes)
    
    def forward(self, observations: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: [batch_size, seq_len, obs_dim]
            context: [batch_size, seq_len, context_dim]
        
        Returns:
            Regime probabilities [batch_size, seq_len, n_regimes]
        """
        # Concatenate observations and context
        x = torch.cat([observations, context], dim=-1)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Compute regime logits and probabilities
        logits = self.output(lstm_out)
        probs = F.softmax(logits, dim=-1)
        
        return probs


# =============================================================================
# Deep Markov Model
# =============================================================================

class DeepMarkovModel:
    """
    Complete Deep Markov Model integrating transition, emission, and inference networks.
    
    Two-Phase Training:
        Phase 1 (Supervised): Train on heuristic regime labels derived from volatility
        Phase 2 (VAE): Fine-tune with variational ELBO objective
    
    Usage:
        model = DeepMarkovModel(regime_names=['calm', 'neutral', 'volatile', 'panic'])
        model.train(historical_data, epochs=100)
        predictions = model.predict(new_data)
    """
    
    def __init__(
        self,
        regime_names: List[str] = None,
        context_dim: int = 3,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        device: str = None
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        self.regime_names = regime_names or ["calm", "neutral", "volatile", "panic"]
        self.n_regimes = len(self.regime_names)
        self.regime_to_idx = {name: i for i, name in enumerate(self.regime_names)}
        self.idx_to_regime = {i: name for i, name in enumerate(self.regime_names)}
        
        # Device configuration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize networks
        self.transition_net = TransitionNetwork(
            n_regimes=self.n_regimes,
            context_dim=context_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.emission_net = EmissionNetwork(
            n_regimes=self.n_regimes,
            context_dim=context_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.inference_net = InferenceNetwork(
            n_regimes=self.n_regimes,
            obs_dim=2,  # (return, volatility)
            context_dim=context_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.transition_net.parameters()) +
            list(self.emission_net.parameters()) +
            list(self.inference_net.parameters()),
            lr=learning_rate
        )
        
        self.training_history = {
            'loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'supervised_loss': [],
            'phase': []
        }
    
    # =========================================================================
    # Data Preparation
    # =========================================================================
    
    def prepare_data(
        self,
        prices: np.ndarray,
        is_tokenized: np.ndarray,
        adoption_rate: Optional[np.ndarray] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare data for training/inference.
        
        Args:
            prices: Price series [batch_size, seq_len]
            is_tokenized: Binary flag [batch_size] (0=traditional, 1=tokenized)
            adoption_rate: Optional adoption rate over time [batch_size, seq_len]
        
        Returns:
            Dictionary with tensors ready for model
        """
        # Calculate log returns
        log_prices = np.log(prices)
        returns = np.diff(log_prices, axis=1)
        
        # Calculate rolling volatility (window=6 periods)
        window = 6
        volatility = np.zeros_like(returns)
        for i in range(returns.shape[1]):
            start = max(0, i - window + 1)
            window_returns = returns[:, start:i+1]
            volatility[:, i] = np.std(window_returns, axis=1)
        
        # Normalize: clip outliers then standardize globally
        returns_clipped = np.clip(returns, -0.3, 0.3)
        returns_mean = np.mean(returns_clipped)
        returns_std = np.std(returns_clipped) + 1e-8
        returns_normalized = (returns_clipped - returns_mean) / returns_std
        
        vol_99 = np.percentile(volatility, 99)
        volatility_clipped = np.clip(volatility, 0, vol_99)
        vol_mean = np.mean(volatility_clipped)
        vol_std = np.std(volatility_clipped) + 1e-8
        volatility_normalized = (volatility_clipped - vol_mean) / vol_std
        
        # Observations: [returns_normalized, volatility_normalized]
        observations = np.stack([returns_normalized, volatility_normalized], axis=-1)
        
        # Context: [is_tokenized, time_normalized, adoption_rate]
        batch_size, seq_len = returns.shape
        time_normalized = np.tile(np.linspace(0, 1, seq_len), (batch_size, 1))
        
        if adoption_rate is None:
            adoption_rate = np.zeros((batch_size, seq_len))
        else:
            if adoption_rate.shape[1] > seq_len:
                adoption_rate = adoption_rate[:, 1:]
        
        is_tokenized_expanded = np.tile(is_tokenized[:, None], (1, seq_len))
        
        context = np.stack([
            is_tokenized_expanded,
            time_normalized,
            adoption_rate
        ], axis=-1)
        
        return {
            'observations': torch.FloatTensor(observations).to(self.device),
            'context': torch.FloatTensor(context).to(self.device),
            'prices': torch.FloatTensor(prices[:, 1:]).to(self.device),
            'returns': returns,
            'volatility': volatility
        }
    
    # =========================================================================
    # Heuristic Regime Labeling
    # =========================================================================
    
    def _assign_heuristic_regimes(
        self,
        returns: np.ndarray,
        volatility: np.ndarray
    ) -> np.ndarray:
        """
        Assign regime labels based on rolling volatility quantiles.
        
        This gives the model a meaningful starting point instead of
        having to discover regimes from scratch.
        
        Args:
            returns: Log returns [batch_size, seq_len]
            volatility: Rolling volatility [batch_size, seq_len]
        
        Returns:
            Regime labels [batch_size, seq_len] with values 0-3
        """
        # Flatten volatility to compute global quantiles
        all_vol = volatility.flatten()
        all_vol = all_vol[all_vol > 0]  # Exclude zeros
        
        if len(all_vol) == 0:
            return np.zeros_like(volatility, dtype=np.int64)
        
        # Use volatility quantiles to define regimes
        # calm: bottom 40%, neutral: 40-70%, volatile: 70-90%, panic: top 10%
        q_calm = np.percentile(all_vol, 40)
        q_neutral = np.percentile(all_vol, 70)
        q_volatile = np.percentile(all_vol, 90)
        
        labels = np.zeros_like(volatility, dtype=np.int64)
        labels[volatility > q_calm] = 1      # neutral
        labels[volatility > q_neutral] = 2   # volatile
        labels[volatility > q_volatile] = 3  # panic
        
        # Also factor in large negative returns for panic
        labels[(returns < -0.05) & (volatility > q_neutral)] = 3  # panic on big drops
        
        return labels
    
    # =========================================================================
    # Phase 1: Supervised Training
    # =========================================================================
    
    def _supervised_step(
        self,
        observations: torch.Tensor,
        context: torch.Tensor,
        regime_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Supervised training step: train inference + emission networks on labels.
        
        Args:
            observations: [batch_size, seq_len, obs_dim]
            context: [batch_size, seq_len, context_dim]
            regime_labels: [batch_size, seq_len] integer labels
        
        Returns:
            Dictionary of loss metrics
        """
        self.optimizer.zero_grad()
        
        batch_size, seq_len, _ = observations.shape
        
        # 1. Inference network: predict regime probabilities
        regime_probs = self.inference_net(observations, context)
        regime_logits = torch.log(regime_probs + 1e-8)
        
        # 2. Classification loss: cross-entropy against heuristic labels
        # Reshape for cross_entropy: [batch*seq, n_regimes] vs [batch*seq]
        logits_flat = regime_logits.view(-1, self.n_regimes)
        labels_flat = regime_labels.view(-1)
        
        classification_loss = F.cross_entropy(logits_flat, labels_flat)
        
        # 3. Emission loss: train emission network to predict observations given labels
        emission_loss = 0.0
        for t in range(seq_len):
            emissions = self.emission_net(regime_labels[:, t], context[:, t])
            return_ll = emissions['return_dist'].log_prob(observations[:, t, 0])
            vol_ll = emissions['vol_dist'].log_prob(observations[:, t, 1])
            emission_loss += (return_ll + vol_ll).mean()
        emission_loss = -emission_loss / seq_len
        
        # 4. Transition loss: train transition network on consecutive labels
        transition_loss = 0.0
        for t in range(seq_len - 1):
            trans_probs = self.transition_net(regime_labels[:, t], context[:, t])
            next_labels_onehot = F.one_hot(regime_labels[:, t + 1], self.n_regimes).float()
            transition_loss += -torch.sum(next_labels_onehot * torch.log(trans_probs + 1e-8), dim=-1).mean()
        transition_loss = transition_loss / max(seq_len - 1, 1)
        
        # 5. Combined supervised loss
        total_loss = classification_loss + emission_loss + transition_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.transition_net.parameters()) +
            list(self.emission_net.parameters()) +
            list(self.inference_net.parameters()),
            max_norm=5.0
        )
        self.optimizer.step()
        
        return {
            'supervised_loss': classification_loss.item(),
            'reconstruction_loss': emission_loss.item(),
            'transition_loss': transition_loss.item(),
            'total_loss': total_loss.item()
        }
    
    # =========================================================================
    # Phase 2: VAE Training
    # =========================================================================
    
    def compute_elbo(
        self,
        observations: torch.Tensor,
        context: torch.Tensor,
        beta: float = 1.0,
        tau: float = 0.67
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Evidence Lower Bound (ELBO) for variational training.
        
        Args:
            observations: [batch_size, seq_len, obs_dim]
            context: [batch_size, seq_len, context_dim]
            beta: KL annealing parameter (0 to 1)
            tau: Temperature for Gumbel-Softmax
        
        Returns:
            tuple: (negative_elbo, metrics_dict)
        """
        batch_size, seq_len, _ = observations.shape
        
        # 1. Inference: q(regime_t | observations)
        regime_probs = self.inference_net(observations, context)
        
        # 2. Sample regimes using Gumbel-Softmax
        regime_logits = torch.log(regime_probs + 1e-8)
        regime_samples = F.gumbel_softmax(regime_logits, tau=tau, hard=True)
        regime_indices = torch.argmax(regime_samples, dim=-1)
        
        # 3. Reconstruction loss
        reconstruction_loss = 0.0
        for t in range(seq_len):
            emissions = self.emission_net(regime_indices[:, t], context[:, t])
            return_ll = emissions['return_dist'].log_prob(observations[:, t, 0])
            vol_ll = emissions['vol_dist'].log_prob(observations[:, t, 1])
            reconstruction_loss += (return_ll + vol_ll).mean()
        reconstruction_loss = -reconstruction_loss / seq_len
        
        # 4. KL divergence
        kl_loss = 0.0
        prior_initial = torch.ones(batch_size, self.n_regimes).to(self.device) / self.n_regimes
        kl_loss += self._kl_categorical(regime_probs[:, 0], prior_initial)
        
        for t in range(seq_len - 1):
            prior_transition = self.transition_net(regime_indices[:, t], context[:, t])
            posterior_transition = regime_probs[:, t + 1]
            kl_loss += self._kl_categorical(posterior_transition, prior_transition)
        
        kl_loss = kl_loss / max(seq_len - 1, 1)
        
        # 5. ELBO
        neg_elbo = reconstruction_loss + beta * kl_loss
        
        metrics = {
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item(),
            'supervised_loss': 0.0,
            'total_loss': neg_elbo.item()
        }
        
        return neg_elbo, metrics
    
    def _kl_categorical(self, q_probs: torch.Tensor, p_probs: torch.Tensor) -> torch.Tensor:
        """KL divergence between categorical distributions."""
        q_probs = q_probs + 1e-8
        p_probs = p_probs + 1e-8
        return (q_probs * (torch.log(q_probs) - torch.log(p_probs))).sum(dim=-1).mean()
    
    def _vae_step(
        self,
        observations: torch.Tensor,
        context: torch.Tensor,
        beta: float = 1.0,
        tau: float = 0.67
    ) -> Dict[str, float]:
        """Single VAE training step."""
        self.optimizer.zero_grad()
        loss, metrics = self.compute_elbo(observations, context, beta, tau)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.transition_net.parameters()) +
            list(self.emission_net.parameters()) +
            list(self.inference_net.parameters()),
            max_norm=5.0
        )
        self.optimizer.step()
        return metrics
    
    # =========================================================================
    # Main Training Loop (Two-Phase)
    # =========================================================================
    
    def train(
        self,
        prices: np.ndarray,
        is_tokenized: np.ndarray,
        adoption_rate: Optional[np.ndarray] = None,
        epochs: int = 200,
        batch_size: int = 32,
        supervised_fraction: float = 0.4,
        beta_schedule: str = 'slow_linear',
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the Deep Markov Model using two-phase approach.
        
        Phase 1 (first supervised_fraction of epochs):
            Supervised training on heuristic regime labels.
            Teaches the model what regimes look like.
            
        Phase 2 (remaining epochs):
            VAE fine-tuning with gradual KL annealing.
            Refines regime boundaries using variational objective.
        
        Args:
            prices: Price series [n_sequences, seq_len]
            is_tokenized: Binary flags [n_sequences]
            adoption_rate: Optional [n_sequences, seq_len]
            epochs: Total number of training epochs
            batch_size: Batch size
            supervised_fraction: Fraction of epochs for Phase 1 (default 0.4)
            beta_schedule: KL schedule for Phase 2
            verbose: Print progress
        
        Returns:
            Training history dictionary
        """
        # Prepare data
        data = self.prepare_data(prices, is_tokenized, adoption_rate)
        observations = data['observations']
        context = data['context']
        
        n_sequences = observations.shape[0]
        
        # Assign heuristic regime labels for supervised phase
        regime_labels = self._assign_heuristic_regimes(data['returns'], data['volatility'])
        regime_labels_tensor = torch.LongTensor(regime_labels).to(self.device)
        
        # Print label statistics
        if verbose:
            unique, counts = np.unique(regime_labels, return_counts=True)
            total = regime_labels.size
            print("\nHeuristic Regime Labels:")
            for u, c in zip(unique, counts):
                name = self.regime_names[u] if u < len(self.regime_names) else f"regime_{u}"
                print(f"  {name:10s}: {c:5d} ({100*c/total:.1f}%)")
        
        supervised_epochs = int(epochs * supervised_fraction)
        vae_epochs = epochs - supervised_epochs
        
        if verbose:
            print(f"\nPhase 1: Supervised ({supervised_epochs} epochs)")
            print(f"Phase 2: VAE fine-tune ({vae_epochs} epochs)")
            print("=" * 70)
        
        # =====================================================================
        # PHASE 1: Supervised Training
        # =====================================================================
        
        for epoch in range(supervised_epochs):
            epoch_metrics = {
                'reconstruction_loss': 0.0,
                'supervised_loss': 0.0,
                'kl_loss': 0.0,
                'total_loss': 0.0
            }
            n_batches = 0
            
            indices = np.random.permutation(n_sequences)
            for i in range(0, n_sequences, batch_size):
                batch_idx = indices[i:i + batch_size]
                
                metrics = self._supervised_step(
                    observations[batch_idx],
                    context[batch_idx],
                    regime_labels_tensor[batch_idx]
                )
                
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key] += metrics[key]
                n_batches += 1
            
            for key in epoch_metrics:
                epoch_metrics[key] /= max(n_batches, 1)
            
            self.training_history['loss'].append(epoch_metrics['total_loss'])
            self.training_history['reconstruction_loss'].append(epoch_metrics['reconstruction_loss'])
            self.training_history['kl_loss'].append(0.0)
            self.training_history['supervised_loss'].append(epoch_metrics['supervised_loss'])
            self.training_history['phase'].append(1)
            
            if verbose and (epoch % 10 == 0 or epoch == supervised_epochs - 1):
                # Check regime usage
                with torch.no_grad():
                    probs = self.inference_net(observations, context)
                    preds = torch.argmax(probs, dim=-1).cpu().numpy()
                    unique, counts = np.unique(preds, return_counts=True)
                    usage = {self.regime_names[u]: c for u, c in zip(unique, counts)}
                
                print(f"[Phase 1] Epoch {epoch:3d}/{supervised_epochs} | "
                      f"Loss: {epoch_metrics['total_loss']:.4f} | "
                      f"Class: {epoch_metrics['supervised_loss']:.4f} | "
                      f"Recon: {epoch_metrics['reconstruction_loss']:.4f} | "
                      f"Usage: {usage}")
        
        if verbose:
            print("\n" + "=" * 70)
            print(f"Phase 2: VAE Fine-tuning ({vae_epochs} epochs)")
            print("=" * 70)
        
        # Lower learning rate for fine-tuning
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.5
        
        # =====================================================================
        # PHASE 2: VAE Fine-tuning
        # =====================================================================
        
        for epoch in range(vae_epochs):
            # KL annealing within Phase 2
            progress = epoch / max(vae_epochs - 1, 1)
            if beta_schedule == 'slow_linear':
                beta = min(1.0, progress * 1.5)  # Reach 1.0 at 67% of Phase 2
            elif beta_schedule == 'linear':
                beta = min(1.0, progress * 2.0)  # Reach 1.0 at 50%
            else:
                beta = 1.0
            
            tau = max(0.5, 1.0 - progress * 0.5)
            
            epoch_metrics = {
                'reconstruction_loss': 0.0,
                'kl_loss': 0.0,
                'supervised_loss': 0.0,
                'total_loss': 0.0
            }
            n_batches = 0
            
            indices = np.random.permutation(n_sequences)
            for i in range(0, n_sequences, batch_size):
                batch_idx = indices[i:i + batch_size]
                
                metrics = self._vae_step(
                    observations[batch_idx],
                    context[batch_idx],
                    beta, tau
                )
                
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key] += metrics[key]
                n_batches += 1
            
            for key in epoch_metrics:
                epoch_metrics[key] /= max(n_batches, 1)
            
            self.training_history['loss'].append(epoch_metrics['total_loss'])
            self.training_history['reconstruction_loss'].append(epoch_metrics['reconstruction_loss'])
            self.training_history['kl_loss'].append(epoch_metrics['kl_loss'])
            self.training_history['supervised_loss'].append(0.0)
            self.training_history['phase'].append(2)
            
            if verbose and (epoch % 10 == 0 or epoch == vae_epochs - 1):
                with torch.no_grad():
                    probs = self.inference_net(observations, context)
                    preds = torch.argmax(probs, dim=-1).cpu().numpy()
                    unique, counts = np.unique(preds, return_counts=True)
                    usage = {self.regime_names[u]: c for u, c in zip(unique, counts)}
                
                print(f"[Phase 2] Epoch {epoch:3d}/{vae_epochs} | "
                      f"Loss: {epoch_metrics['total_loss']:.4f} | "
                      f"Recon: {epoch_metrics['reconstruction_loss']:.4f} | "
                      f"KL: {epoch_metrics['kl_loss']:.4f} | "
                      f"Beta: {beta:.3f} | "
                      f"Usage: {usage}")
        
        return self.training_history
    
    # =========================================================================
    # Inference
    # =========================================================================
    
    def infer_regimes(
        self,
        prices: np.ndarray,
        is_tokenized: float,
        adoption_rate: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Infer regime sequence from price data.
        
        Args:
            prices: Price series [seq_len]
            is_tokenized: 0 for traditional, 1 for tokenized
            adoption_rate: Optional adoption rates [seq_len]
        
        Returns:
            tuple: (regime_names, regime_probabilities)
        """
        prices_batch = prices[None, :]
        is_tokenized_batch = np.array([is_tokenized])
        
        if adoption_rate is not None:
            adoption_rate = adoption_rate[None, :]
        
        data = self.prepare_data(prices_batch, is_tokenized_batch, adoption_rate)
        
        with torch.no_grad():
            regime_probs = self.inference_net(
                data['observations'],
                data['context']
            )[0].cpu().numpy()
        
        regime_indices = np.argmax(regime_probs, axis=-1)
        regime_names = [self.idx_to_regime[idx] for idx in regime_indices]
        
        return np.array(regime_names), regime_probs
    
    def predict_next_regime(
        self,
        current_regime: str,
        context: Dict[str, float]
    ) -> Tuple[str, np.ndarray]:
        """
        Predict next regime given current regime and context.
        
        Args:
            current_regime: Current regime name
            context: Dict with 'is_tokenized', 'time_normalized', 'adoption_rate'
        
        Returns:
            tuple: (predicted_regime_name, transition_probabilities)
        """
        regime_idx = torch.LongTensor([self.regime_to_idx[current_regime]]).to(self.device)
        
        context_tensor = torch.FloatTensor([[
            context['is_tokenized'],
            context['time_normalized'],
            context.get('adoption_rate', 0.0)
        ]]).to(self.device)
        
        with torch.no_grad():
            probs = self.transition_net(regime_idx, context_tensor)[0].cpu().numpy()
        
        next_regime_idx = np.argmax(probs)
        next_regime = self.idx_to_regime[next_regime_idx]
        
        return next_regime, probs
    
    # =========================================================================
    # Save / Load
    # =========================================================================
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'transition_net': self.transition_net.state_dict(),
            'emission_net': self.emission_net.state_dict(),
            'inference_net': self.inference_net.state_dict(),
            'regime_names': self.regime_names,
            'training_history': self.training_history
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.transition_net.load_state_dict(checkpoint['transition_net'])
        self.emission_net.load_state_dict(checkpoint['emission_net'])
        self.inference_net.load_state_dict(checkpoint['inference_net'])
        self.regime_names = checkpoint['regime_names']
        self.training_history = checkpoint.get('training_history', {})
        print(f"Model loaded from {path}")
