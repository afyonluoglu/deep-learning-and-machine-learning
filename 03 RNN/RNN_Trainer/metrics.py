"""
Advanced Metrics and Monitoring for RNN Training
Includes gradient monitoring, weight analysis, and comprehensive metrics
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque


class GradientMonitor:
    """Monitor gradients for vanishing/exploding gradient detection."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize gradient monitor.
        
        Args:
            window_size: Number of recent gradients to track
        """
        self.window_size = window_size
        self.grad_norms = {
            'Wxh': deque(maxlen=window_size),
            'Whh': deque(maxlen=window_size),
            'Why': deque(maxlen=window_size),
            'bh': deque(maxlen=window_size),
            'by': deque(maxlen=window_size)
        }
        self.total_norms = deque(maxlen=window_size)
        self.warnings = []
    
    def track(self, grads: Tuple[np.ndarray, ...]) -> Dict[str, Any]:
        """
        Track gradient norms.
        
        Args:
            grads: Tuple of (dWxh, dWhh, dWhy, dbh, dby)
        
        Returns:
            Dictionary with gradient statistics
        """
        dWxh, dWhh, dWhy, dbh, dby = grads
        
        # Compute norms
        norm_wxh = np.linalg.norm(dWxh)
        norm_whh = np.linalg.norm(dWhh)
        norm_why = np.linalg.norm(dWhy)
        norm_bh = np.linalg.norm(dbh)
        norm_by = np.linalg.norm(dby)
        
        total_norm = norm_wxh + norm_whh + norm_why + norm_bh + norm_by
        
        # Store
        self.grad_norms['Wxh'].append(norm_wxh)
        self.grad_norms['Whh'].append(norm_whh)
        self.grad_norms['Why'].append(norm_why)
        self.grad_norms['bh'].append(norm_bh)
        self.grad_norms['by'].append(norm_by)
        self.total_norms.append(total_norm)
        
        # Detect problems
        status = "HEALTHY"
        warning = None
        
        if total_norm < 0.0001:
            status = "VANISHING"
            warning = "⚠️ Vanishing gradient detected! Consider using LSTM/GRU or gradient clipping."
            self.warnings.append(warning)
        elif total_norm > 100:
            status = "EXPLODING"
            warning = "⚠️ Exploding gradient detected! Reduce learning rate or increase clipping threshold."
            self.warnings.append(warning)
        
        return {
            'norms': {
                'Wxh': norm_wxh,
                'Whh': norm_whh,
                'Why': norm_why,
                'bh': norm_bh,
                'by': norm_by
            },
            'total_norm': total_norm,
            'status': status,
            'warning': warning,
            'avg_total_norm': np.mean(self.total_norms) if self.total_norms else 0.0,
            'max_layer': max(self.grad_norms, key=lambda k: self.grad_norms[k][-1] if self.grad_norms[k] else 0)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get gradient statistics over window."""
        stats = {}
        
        for key, norms in self.grad_norms.items():
            if norms:
                stats[key] = {
                    'mean': np.mean(norms),
                    'std': np.std(norms),
                    'min': np.min(norms),
                    'max': np.max(norms),
                    'recent': norms[-1]
                }
        
        if self.total_norms:
            stats['total'] = {
                'mean': np.mean(self.total_norms),
                'std': np.std(self.total_norms),
                'min': np.min(self.total_norms),
                'max': np.max(self.total_norms),
                'recent': self.total_norms[-1]
            }
        
        return stats


class WeightAnalyzer:
    """Analyze weight distributions and detect dead neurons."""
    
    def __init__(self):
        """Initialize weight analyzer."""
        self.weight_history = {
            'Wxh': [],
            'Whh': [],
            'Why': [],
            'bh': [],
            'by': []
        }
    
    def analyze(self, weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze weight distributions.
        
        Args:
            weights: Dictionary of weight matrices
        
        Returns:
            Analysis results
        """
        results = {}
        
        for key, W in weights.items():
            # Basic statistics
            results[key] = {
                'mean': np.mean(W),
                'std': np.std(W),
                'min': np.min(W),
                'max': np.max(W),
                'abs_mean': np.mean(np.abs(W)),
                'sparsity': np.sum(np.abs(W) < 1e-6) / W.size,  # Fraction of near-zero weights
            }
            
            # Dead neurons (for weight matrices)
            if len(W.shape) == 2:
                # Row-wise (output neurons)
                row_norms = np.linalg.norm(W, axis=1)
                dead_outputs = np.sum(row_norms < 1e-6)
                results[key]['dead_outputs'] = dead_outputs
                results[key]['dead_output_ratio'] = dead_outputs / W.shape[0]
                
                # Column-wise (input neurons)
                col_norms = np.linalg.norm(W, axis=0)
                dead_inputs = np.sum(col_norms < 1e-6)
                results[key]['dead_inputs'] = dead_inputs
                results[key]['dead_input_ratio'] = dead_inputs / W.shape[1]
            
            # Store history
            self.weight_history[key].append({
                'mean': results[key]['mean'],
                'std': results[key]['std']
            })
        
        return results
    
    def get_weight_histogram(self, weights: np.ndarray, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get histogram of weight values.
        
        Args:
            weights: Weight matrix
            bins: Number of bins
        
        Returns:
            (counts, bin_edges)
        """
        return np.histogram(weights.flatten(), bins=bins)


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics."""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate all regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Dictionary of metrics
        """
        # Ensure same shape
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # MSE (Mean Squared Error)
        mse = np.mean((y_true - y_pred) ** 2)
        
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mse)
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # MAPE (Mean Absolute Percentage Error) - avoid division by zero
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        # R² Score (Coefficient of Determination)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        # Max Error
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Median Absolute Error
        median_ae = np.median(np.abs(y_true - y_pred))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'max_error': max_error,
            'median_ae': median_ae
        }
    
    @staticmethod
    def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (for time series).
        Percentage of correct direction predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Directional accuracy (0-100%)
        """
        if len(y_true) < 2:
            return 0.0
        
        true_diff = np.diff(y_true.flatten())
        pred_diff = np.diff(y_pred.flatten())
        
        # Same sign means correct direction
        correct = np.sum(np.sign(true_diff) == np.sign(pred_diff))
        total = len(true_diff)
        
        return (correct / total) * 100 if total > 0 else 0.0


class TrainingMonitor:
    """Monitor training progress and detect convergence issues."""
    
    def __init__(self, patience: int = 10):
        """
        Initialize training monitor.
        
        Args:
            patience: Epochs to wait for improvement before early stopping
        """
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.loss_history = []
        self.plateau_detected = False
    
    def update(self, loss: float) -> Dict[str, Any]:
        """
        Update monitor with new loss value.
        
        Args:
            loss: Current loss value
        
        Returns:
            Monitoring status
        """
        self.loss_history.append(loss)
        
        # Check for improvement
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
            improvement = True
        else:
            self.patience_counter += 1
            improvement = False
        
        # Detect plateau (last N losses are similar)
        if len(self.loss_history) >= 5:
            recent_losses = self.loss_history[-5:]
            if np.std(recent_losses) < 1e-6:
                self.plateau_detected = True
            else:
                self.plateau_detected = False
        
        # Calculate loss improvement rate
        if len(self.loss_history) >= 2:
            improvement_rate = (self.loss_history[-2] - self.loss_history[-1]) / (self.loss_history[-2] + 1e-10)
        else:
            improvement_rate = 0.0
        
        return {
            'current_loss': loss,
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter,
            'should_stop': self.patience_counter >= self.patience,
            'plateau_detected': self.plateau_detected,
            'improvement': improvement,
            'improvement_rate': improvement_rate * 100  # Percentage
        }
    
    def get_convergence_score(self) -> float:
        """
        Calculate convergence score (0-100).
        Higher score = better convergence.
        
        Returns:
            Convergence score
        """
        if len(self.loss_history) < 5:
            return 0.0
        
        # Calculate smoothness (lower std = smoother = better)
        recent_losses = self.loss_history[-10:]
        smoothness = 100 / (1 + np.std(recent_losses))
        
        # Calculate improvement (more improvement = better)
        if len(self.loss_history) >= 10:
            initial_avg = np.mean(self.loss_history[:5])
            recent_avg = np.mean(self.loss_history[-5:])
            improvement = ((initial_avg - recent_avg) / (initial_avg + 1e-10)) * 100
        else:
            improvement = 0.0
        
        # Combine metrics
        score = (smoothness + improvement) / 2
        return np.clip(score, 0, 100)
