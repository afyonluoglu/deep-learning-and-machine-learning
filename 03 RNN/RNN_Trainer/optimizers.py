"""
Advanced Optimizers for RNN Training
Supports: SGD, SGD+Momentum, Adam, RMSprop
"""
import numpy as np
from typing import Dict, Any


class Optimizer:
    """Base optimizer class."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.t = 0  # Timestep counter
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Update parameters using gradients."""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent."""
    
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        updated_params = {}
        for key in params:
            updated_params[key] = params[key] - self.learning_rate * grads[key]
        return updated_params


class SGDMomentum(Optimizer):
    """SGD with Momentum."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        updated_params = {}
        
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            
            # Update velocity
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            
            # Update parameters
            updated_params[key] = params[key] + self.velocity[key]
        
        return updated_params


class Adam(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation)."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment (mean)
        self.v = {}  # Second moment (variance)
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        updated_params = {}
        self.t += 1
        
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            updated_params[key] = params[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_params


class RMSprop(Optimizer):
    """RMSprop optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.cache = {}
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        updated_params = {}
        
        for key in params:
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            
            # Update cache (moving average of squared gradients)
            self.cache[key] = self.beta * self.cache[key] + (1 - self.beta) * (grads[key] ** 2)
            
            # Update parameters
            updated_params[key] = params[key] - self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)
        
        return updated_params


class LearningRateScheduler:
    """Learning rate scheduling strategies."""
    
    def __init__(self, initial_lr: float, schedule_type: str = 'constant', **kwargs):
        """
        Initialize scheduler.
        
        Args:
            initial_lr: Initial learning rate
            schedule_type: Type of schedule ('constant', 'step', 'exponential', 'cosine', 
                          'reduce_on_plateau', 'cyclical', 'warmup_decay')
            **kwargs: Additional parameters for specific schedulers
                - step_size: Steps before decay (for step)
                - gamma: Decay factor (for step and exponential)
                - T_max: Maximum epochs (for cosine)
                - eta_min: Minimum learning rate (for cosine)
                - patience: Epochs to wait before reducing (for reduce_on_plateau)
                - factor: Factor to reduce LR (for reduce_on_plateau)
                - base_lr: Base learning rate (for cyclical)
                - max_lr: Maximum learning rate (for cyclical)
                - step_size_cycle: Steps in half cycle (for cyclical)
                - warmup_steps: Number of warmup steps (for warmup_decay)
                - decay_steps: Number of decay steps (for warmup_decay)
        """
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type
        self.current_epoch = 0
        self.kwargs = kwargs
        
        # For reduce_on_plateau
        self.best_loss = float('inf')
        self.wait = 0
        self.current_lr = initial_lr
        
        # For cyclical LR
        self.cycle_step = 0
    
    def get_lr(self, epoch: int = None, loss: float = None) -> float:
        """
        Get learning rate for current epoch.
        
        Args:
            epoch: Current epoch number (optional)
            loss: Current loss value (for reduce_on_plateau)
        
        Returns:
            Current learning rate
        """
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.schedule_type == 'constant':
            return self.initial_lr
        
        elif self.schedule_type == 'step':
            step_size = self.kwargs.get('step_size', 10)
            gamma = self.kwargs.get('gamma', 0.5)
            return self.initial_lr * (gamma ** (self.current_epoch // step_size))
        
        elif self.schedule_type == 'exponential':
            gamma = self.kwargs.get('gamma', 0.95)
            return self.initial_lr * (gamma ** self.current_epoch)
        
        elif self.schedule_type == 'cosine':
            T_max = self.kwargs.get('T_max', 100)
            eta_min = self.kwargs.get('eta_min', 0.0001)
            return eta_min + (self.initial_lr - eta_min) * (1 + np.cos(np.pi * self.current_epoch / T_max)) / 2
        
        elif self.schedule_type == 'reduce_on_plateau':
            # Reduce LR when loss plateaus
            patience = self.kwargs.get('patience', 10)
            factor = self.kwargs.get('factor', 0.5)
            min_lr = self.kwargs.get('min_lr', 1e-6)
            
            if loss is not None:
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= patience:
                        self.current_lr = max(self.current_lr * factor, min_lr)
                        self.wait = 0
            
            return self.current_lr
        
        elif self.schedule_type == 'cyclical':
            # Cyclical Learning Rate (CLR)
            base_lr = self.kwargs.get('base_lr', self.initial_lr * 0.1)
            max_lr = self.kwargs.get('max_lr', self.initial_lr)
            step_size_cycle = self.kwargs.get('step_size_cycle', 2000)
            
            cycle = np.floor(1 + self.cycle_step / (2 * step_size_cycle))
            x = np.abs(self.cycle_step / step_size_cycle - 2 * cycle + 1)
            lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
            
            return lr
        
        elif self.schedule_type == 'warmup_decay':
            # Warmup + Decay schedule
            warmup_steps = self.kwargs.get('warmup_steps', 1000)
            decay_steps = self.kwargs.get('decay_steps', 10000)
            min_lr = self.kwargs.get('min_lr', 1e-6)
            
            if self.current_epoch < warmup_steps:
                # Linear warmup
                lr = self.initial_lr * (self.current_epoch + 1) / warmup_steps
            else:
                # Exponential decay after warmup
                decay_rate = (min_lr / self.initial_lr) ** (1 / decay_steps)
                steps_after_warmup = self.current_epoch - warmup_steps
                lr = self.initial_lr * (decay_rate ** steps_after_warmup)
                lr = max(lr, min_lr)
            
            return lr
        
        return self.initial_lr
    
    def step(self, loss: float = None):
        """
        Increment epoch counter.
        
        Args:
            loss: Current loss value (for reduce_on_plateau)
        """
        self.current_epoch += 1
        self.cycle_step += 1  # For cyclical LR


def create_optimizer(optimizer_type: str, learning_rate: float, **kwargs) -> Optimizer:
    """
    Factory function to create optimizer.
    
    Args:
        optimizer_type: Type of optimizer ('sgd', 'momentum', 'adam', 'rmsprop')
        learning_rate: Learning rate
        **kwargs: Additional optimizer-specific parameters
    
    Returns:
        Optimizer instance
    """
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'sgd':
        return SGD(learning_rate)
    
    elif optimizer_type == 'momentum':
        momentum = kwargs.get('momentum', 0.9)
        return SGDMomentum(learning_rate, momentum)
    
    elif optimizer_type == 'adam':
        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.999)
        epsilon = kwargs.get('epsilon', 1e-8)
        return Adam(learning_rate, beta1, beta2, epsilon)
    
    elif optimizer_type == 'rmsprop':
        beta = kwargs.get('beta', 0.9)
        epsilon = kwargs.get('epsilon', 1e-8)
        return RMSprop(learning_rate, beta, epsilon)
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
