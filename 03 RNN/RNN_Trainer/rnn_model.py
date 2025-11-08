"""
RNN Model Implementation with Backpropagation Through Time (BPTT)
Enhanced with advanced optimizers and metrics
"""
import numpy as np
import pickle
from typing import Tuple, List, Dict, Any, Optional

# Import advanced modules
try:
    from optimizers import create_optimizer, LearningRateScheduler
    from metrics import GradientMonitor, WeightAnalyzer, MetricsCalculator, TrainingMonitor
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False
    print("⚠️ Advanced features not available. Using basic optimization.")


class RNNModel:
    """
    Recurrent Neural Network with backpropagation through time.
    """
    
    def __init__(self, 
                 input_size: int = 1,
                 hidden_size: int = 10,
                 output_size: int = 1,
                 learning_rate: float = 0.01,
                 sequence_length: int = 10,
                 activation: str = 'tanh',
                 dropout_rate: float = 0.0,
                 optimizer_type: str = 'sgd',
                 lr_schedule: str = 'constant',
                 num_layers: int = 1,
                 hidden_sizes: List[int] = None,
                 **optimizer_kwargs):
        """
        Initialize RNN model (supports multi-layer/stacked RNN).
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units (used if hidden_sizes not provided)
            output_size: Number of output features
            learning_rate: Learning rate for gradient descent
            sequence_length: Length of sequences for BPTT
            activation: Activation function ('tanh' or 'relu')
            dropout_rate: Dropout rate for regularization (0.0 to 0.9)
            optimizer_type: Type of optimizer ('sgd', 'momentum', 'adam', 'rmsprop')
            lr_schedule: Learning rate schedule ('constant', 'step', 'exponential', 'cosine', 
                        'reduce_on_plateau', 'cyclical', 'warmup_decay')
            num_layers: Number of hidden layers (1 for single, 2+ for stacked/deep RNN)
            hidden_sizes: List of hidden sizes for each layer (e.g., [20, 15, 10])
                         If None, all layers use hidden_size
            **optimizer_kwargs: Additional optimizer parameters
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.activation_type = activation
        self.dropout_rate = dropout_rate
        self.optimizer_type = optimizer_type
        self.lr_schedule_type = lr_schedule
        self.training_mode = True  # Flag for dropout during training vs prediction
        
        # Multi-layer configuration
        self.num_layers = max(1, num_layers)
        if hidden_sizes is None:
            self.hidden_sizes = [hidden_size] * self.num_layers
        else:
            self.hidden_sizes = hidden_sizes[:self.num_layers] if len(hidden_sizes) >= self.num_layers else hidden_sizes + [hidden_size] * (self.num_layers - len(hidden_sizes))
        
        # Initialize weights with Xavier initialization
        # For multi-layer RNN, we need weights for each layer
        self.Wxh_layers = []  # Input to hidden weights for each layer
        self.Whh_layers = []  # Hidden to hidden weights for each layer
        self.bh_layers = []   # Biases for each layer
        
        for layer in range(self.num_layers):
            if layer == 0:
                # First layer: input_size -> hidden_sizes[0]
                Wxh = np.random.randn(self.hidden_sizes[0], input_size) * np.sqrt(2.0 / input_size)
                Whh = np.random.randn(self.hidden_sizes[0], self.hidden_sizes[0]) * np.sqrt(2.0 / self.hidden_sizes[0])
                bh = np.zeros((self.hidden_sizes[0], 1))
            else:
                # Subsequent layers: hidden_sizes[layer-1] -> hidden_sizes[layer]
                Wxh = np.random.randn(self.hidden_sizes[layer], self.hidden_sizes[layer-1]) * np.sqrt(2.0 / self.hidden_sizes[layer-1])
                Whh = np.random.randn(self.hidden_sizes[layer], self.hidden_sizes[layer]) * np.sqrt(2.0 / self.hidden_sizes[layer])
                bh = np.zeros((self.hidden_sizes[layer], 1))
            
            self.Wxh_layers.append(Wxh)
            self.Whh_layers.append(Whh)
            self.bh_layers.append(bh)
        
        # Output layer weights: last hidden layer -> output
        self.Why = np.random.randn(output_size, self.hidden_sizes[-1]) * np.sqrt(2.0 / self.hidden_sizes[-1])
        self.by = np.zeros((output_size, 1))
        
        # Keep old single-layer variables for backward compatibility
        self.Wxh = self.Wxh_layers[0]
        self.Whh = self.Whh_layers[0]
        self.bh = self.bh_layers[0]
        
        # Training history
        self.loss_history = []
        self.epoch_losses = []
        
        # Advanced features (if available)
        if ADVANCED_FEATURES:
            # Initialize optimizer
            self.optimizer = create_optimizer(optimizer_type, learning_rate, **optimizer_kwargs)
            
            # Initialize LR scheduler
            self.lr_scheduler = LearningRateScheduler(learning_rate, lr_schedule, **optimizer_kwargs)
            
            # Initialize monitors
            self.gradient_monitor = GradientMonitor(window_size=100)
            self.weight_analyzer = WeightAnalyzer()
            self.training_monitor = TrainingMonitor(patience=20)
            self.metrics_calculator = MetricsCalculator()
            
            # Advanced metrics storage
            self.gradient_stats_history = []
            self.weight_stats_history = []
            self.comprehensive_metrics = []
        else:
            self.optimizer = None
            self.lr_scheduler = None
            self.gradient_monitor = None
            self.weight_analyzer = None
            self.training_monitor = None
            self.metrics_calculator = None
        
    def activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation_type == 'tanh':
            return np.tanh(x)
        elif self.activation_type == 'relu':
            return np.maximum(0, x)
        return x
    
    def activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculate derivative of activation function."""
        if self.activation_type == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_type == 'relu':
            return (x > 0).astype(float)
        return np.ones_like(x)
    
    def forward(self, inputs: np.ndarray, h_prev: List[np.ndarray]) -> Tuple[Dict, List[Dict], List[Dict], Dict, Dict]:
        """
        Forward pass through the multi-layer RNN.
        
        Args:
            inputs: Input sequence [seq_len, input_size]
            h_prev: List of previous hidden states for each layer
            
        Returns:
            outputs, hidden states for each layer, and pre-activation values
        """
        xs, ys, ps = {}, {}, {}
        # For multi-layer: hs_layers[layer][t] = hidden state of layer 'layer' at time 't'
        hs_layers = [{} for _ in range(self.num_layers)]
        hs_raw_layers = [{} for _ in range(self.num_layers)]
        dropout_masks_layers = [{} for _ in range(self.num_layers)]
        
        # Initialize previous hidden states
        for layer in range(self.num_layers):
            hs_layers[layer][-1] = np.copy(h_prev[layer])
        
        # Forward pass through time
        for t in range(len(inputs)):
            xs[t] = inputs[t].reshape(-1, 1)
            
            # Process each layer
            for layer in range(self.num_layers):
                if layer == 0:
                    # First layer: use input
                    layer_input = xs[t]
                else:
                    # Subsequent layers: use previous layer's hidden state
                    layer_input = hs_layers[layer-1][t]
                
                # RNN computation for this layer
                hs_raw_layers[layer][t] = (np.dot(self.Wxh_layers[layer], layer_input) + 
                                          np.dot(self.Whh_layers[layer], hs_layers[layer][t-1]) + 
                                          self.bh_layers[layer])
                hs_layers[layer][t] = self.activation(hs_raw_layers[layer][t])
                
                # Apply dropout during training (after each layer except the last)
                if self.training_mode and self.dropout_rate > 0 and layer < self.num_layers - 1:
                    dropout_masks_layers[layer][t] = (np.random.rand(*hs_layers[layer][t].shape) > self.dropout_rate).astype(float)
                    # Inverted dropout: scale by 1/(1-p) during training
                    hs_layers[layer][t] = hs_layers[layer][t] * dropout_masks_layers[layer][t] / (1 - self.dropout_rate)
            
            # Output layer (uses the last hidden layer)
            ys[t] = np.dot(self.Why, hs_layers[-1][t]) + self.by
            ps[t] = ys[t]  # Linear output for regression
        
        return xs, hs_layers, hs_raw_layers, ys, ps
    
    def backward(self, 
                 xs: Dict, 
                 hs_layers: List[Dict], 
                 hs_raw_layers: List[Dict], 
                 ps: Dict, 
                 targets: np.ndarray) -> Tuple[float, Tuple]:
        """
        Backward pass - Backpropagation Through Time (BPTT) for multi-layer RNN.
        
        Args:
            xs: Input states
            hs_layers: Hidden states for each layer
            hs_raw_layers: Pre-activation hidden states for each layer
            ps: Predictions
            targets: Target values
            
        Returns:
            loss and gradients
        """
        # Initialize gradients for each layer
        dWxh_layers = [np.zeros_like(W) for W in self.Wxh_layers]
        dWhh_layers = [np.zeros_like(W) for W in self.Whh_layers]
        dbh_layers = [np.zeros_like(b) for b in self.bh_layers]
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)
        
        # Initialize dhnext for each layer
        dhnext_layers = [np.zeros_like(hs_layers[layer][0]) for layer in range(self.num_layers)]
        
        loss = 0
        
        # Backward pass through time
        for t in reversed(range(len(targets))):
            target = targets[t].reshape(-1, 1)
            
            # Output layer gradient
            dy = ps[t] - target
            loss += 0.5 * np.sum(dy ** 2)
            
            dWhy += np.dot(dy, hs_layers[-1][t].T)
            dby += dy
            
            # Gradient flowing into the last hidden layer
            dh_from_output = np.dot(self.Why.T, dy)
            
            # Backpropagate through layers (from last to first)
            for layer in reversed(range(self.num_layers)):
                # Combine gradient from output (if last layer) and from next layer in time
                if layer == self.num_layers - 1:
                    dh = dh_from_output + dhnext_layers[layer]
                else:
                    dh = dh_from_next_layer + dhnext_layers[layer]
                
                # Gradient through activation
                dhraw = dh * self.activation_derivative(hs_raw_layers[layer][t])
                
                # Accumulate gradients
                dbh_layers[layer] += dhraw
                
                if layer == 0:
                    # First layer: gradient w.r.t. input
                    dWxh_layers[layer] += np.dot(dhraw, xs[t].T)
                else:
                    # Subsequent layers: gradient w.r.t. previous layer's hidden state
                    dWxh_layers[layer] += np.dot(dhraw, hs_layers[layer-1][t].T)
                    # Pass gradient to previous layer
                    dh_from_next_layer = np.dot(self.Wxh_layers[layer].T, dhraw)
                
                dWhh_layers[layer] += np.dot(dhraw, hs_layers[layer][t-1].T)
                
                # Gradient for next time step
                dhnext_layers[layer] = np.dot(self.Whh_layers[layer].T, dhraw)
        
        # Clip gradients to prevent exploding gradients
        all_grads = dWxh_layers + dWhh_layers + dbh_layers + [dWhy, dby]
        for dparam in all_grads:
            np.clip(dparam, -5, 5, out=dparam)
        
        # Track gradients (if monitor available)
        if self.gradient_monitor is not None:
            # For backward compatibility, track first layer gradients
            grad_stats = self.gradient_monitor.track((dWxh_layers[0], dWhh_layers[0], dWhy, dbh_layers[0], dby))
            self.gradient_stats_history.append(grad_stats)
        
        return loss, (dWxh_layers, dWhh_layers, dWhy, dbh_layers, dby)
    
    def update_weights(self, grads: Tuple) -> None:
        """Update weights using optimizer or basic gradient descent (multi-layer support)."""
        dWxh_layers, dWhh_layers, dWhy, dbh_layers, dby = grads
        
        if self.optimizer is not None:
            # Use advanced optimizer
            params = {'Why': self.Why, 'by': self.by}
            grad_dict = {'Why': dWhy, 'by': dby}
            
            # Add parameters for each layer
            for layer in range(self.num_layers):
                params[f'Wxh_{layer}'] = self.Wxh_layers[layer]
                params[f'Whh_{layer}'] = self.Whh_layers[layer]
                params[f'bh_{layer}'] = self.bh_layers[layer]
                
                grad_dict[f'Wxh_{layer}'] = dWxh_layers[layer]
                grad_dict[f'Whh_{layer}'] = dWhh_layers[layer]
                grad_dict[f'bh_{layer}'] = dbh_layers[layer]
            
            updated_params = self.optimizer.update(params, grad_dict)
            
            # Update weights
            self.Why = updated_params['Why']
            self.by = updated_params['by']
            
            for layer in range(self.num_layers):
                self.Wxh_layers[layer] = updated_params[f'Wxh_{layer}']
                self.Whh_layers[layer] = updated_params[f'Whh_{layer}']
                self.bh_layers[layer] = updated_params[f'bh_{layer}']
        else:
            # Basic gradient descent
            self.Why -= self.learning_rate * dWhy
            self.by -= self.learning_rate * dby
            
            for layer in range(self.num_layers):
                self.Wxh_layers[layer] -= self.learning_rate * dWxh_layers[layer]
                self.Whh_layers[layer] -= self.learning_rate * dWhh_layers[layer]
                self.bh_layers[layer] -= self.learning_rate * dbh_layers[layer]
        
        # Update backward compatibility variables
        self.Wxh = self.Wxh_layers[0]
        self.Whh = self.Whh_layers[0]
        self.bh = self.bh_layers[0]
    
    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Train for one epoch (multi-layer support).
        
        Args:
            X: Input data [n_samples, input_size]
            y: Target data [n_samples, output_size]
            
        Returns:
            Average loss for the epoch
        """
        self.training_mode = True  # Enable dropout
        total_loss = 0
        n_batches = 0
        
        # Initialize hidden states for all layers
        h_prev = [np.zeros((self.hidden_sizes[layer], 1)) for layer in range(self.num_layers)]
        
        # Process sequences
        for i in range(0, len(X) - self.sequence_length, self.sequence_length):
            inputs = X[i:i + self.sequence_length]
            targets = y[i:i + self.sequence_length]
            
            # Forward pass
            xs, hs_layers, hs_raw_layers, ys, ps = self.forward(inputs, h_prev)
            
            # Backward pass
            loss, grads = self.backward(xs, hs_layers, hs_raw_layers, ps, targets)
            
            # Update weights
            self.update_weights(grads)
            
            total_loss += loss
            n_batches += 1
            
            # Update hidden states for next sequence (use last time step from each layer)
            for layer in range(self.num_layers):
                h_prev[layer] = hs_layers[layer][len(inputs) - 1]
            
            # Store loss
            self.loss_history.append(loss / self.sequence_length)
        
        avg_loss = total_loss / max(n_batches, 1)
        self.epoch_losses.append(avg_loss)
        
        # Update learning rate schedule (if available)
        if self.lr_scheduler is not None:
            self.optimizer.learning_rate = self.lr_scheduler.get_lr(loss=avg_loss)
            self.lr_scheduler.step(loss=avg_loss)
        
        # Analyze weights (if available)
        if self.weight_analyzer is not None:
            weight_stats = self.weight_analyzer.analyze({
                'Wxh': self.Wxh,
                'Whh': self.Whh,
                'Why': self.Why,
                'bh': self.bh,
                'by': self.by
            })
            self.weight_stats_history.append(weight_stats)
        
        # Monitor training progress (if available)
        if self.training_monitor is not None:
            self.training_monitor.update(avg_loss)
        
        return avg_loss
    
    def predict(self, X: np.ndarray, h_init: List[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions (multi-layer support).
        
        Args:
            X: Input data
            h_init: List of initial hidden states for each layer
            
        Returns:
            Predictions
        """
        self.training_mode = False  # Disable dropout
        
        if h_init is None:
            h_states = [np.zeros((self.hidden_sizes[layer], 1)) for layer in range(self.num_layers)]
        else:
            h_states = h_init
            
        predictions = []
        
        for i in range(len(X)):
            x = X[i].reshape(-1, 1)
            
            # Forward through all layers
            for layer in range(self.num_layers):
                if layer == 0:
                    layer_input = x
                else:
                    layer_input = h_states[layer-1]
                
                h_raw = (np.dot(self.Wxh_layers[layer], layer_input) + 
                        np.dot(self.Whh_layers[layer], h_states[layer]) + 
                        self.bh_layers[layer])
                h_states[layer] = self.activation(h_raw)
            
            # Output from last layer
            y = np.dot(self.Why, h_states[-1]) + self.by
            predictions.append(y.flatten())
        
        return np.array(predictions)
    
    def predict_sequence(self, seed: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Generate a sequence by predicting future values (multi-layer support).
        
        Args:
            seed: Initial input values
            n_steps: Number of steps to predict
            
        Returns:
            Predicted sequence
        """
        self.training_mode = False  # Disable dropout
        h_states = [np.zeros((self.hidden_sizes[layer], 1)) for layer in range(self.num_layers)]
        predictions = []
        
        # Warm up with seed
        for i in range(len(seed)):
            x = seed[i].reshape(-1, 1)
            
            # Forward through all layers
            for layer in range(self.num_layers):
                if layer == 0:
                    layer_input = x
                else:
                    layer_input = h_states[layer-1]
                
                h_raw = (np.dot(self.Wxh_layers[layer], layer_input) + 
                        np.dot(self.Whh_layers[layer], h_states[layer]) + 
                        self.bh_layers[layer])
                h_states[layer] = self.activation(h_raw)
            
            y = np.dot(self.Why, h_states[-1]) + self.by
            predictions.append(y.flatten())
        
        # Generate future predictions
        last_input = predictions[-1] if predictions else seed[-1]
        for _ in range(n_steps):
            x = last_input.reshape(-1, 1)
            
            # Forward through all layers
            for layer in range(self.num_layers):
                if layer == 0:
                    layer_input = x
                else:
                    layer_input = h_states[layer-1]
                
                h_raw = (np.dot(self.Wxh_layers[layer], layer_input) + 
                        np.dot(self.Whh_layers[layer], h_states[layer]) + 
                        self.bh_layers[layer])
                h_states[layer] = self.activation(h_raw)
            
            y = np.dot(self.Why, h_states[-1]) + self.by
            predictions.append(y.flatten())
            last_input = y.flatten()
        
        return np.array(predictions)
    
    def get_comprehensive_metrics(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            X: Input data
            y_true: True values
        
        Returns:
            Dictionary of metrics
        """
        # Make predictions
        y_pred = self.predict(X)
        
        if self.metrics_calculator is not None:
            return self.metrics_calculator.calculate_all_metrics(y_true, y_pred)
        else:
            # Basic MSE only
            mse = np.mean((y_true.flatten() - y_pred.flatten()) ** 2)
            return {'mse': mse}
    
    def get_gradient_health(self) -> Dict[str, Any]:
        """Get gradient health statistics."""
        if self.gradient_monitor is not None:
            return self.gradient_monitor.get_statistics()
        return {}
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        """Get current weight statistics."""
        if self.weight_analyzer is not None:
            return self.weight_analyzer.analyze({
                'Wxh': self.Wxh,
                'Whh': self.Whh,
                'Why': self.Why,
                'bh': self.bh,
                'by': self.by
            })
        return {}
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get training monitor status."""
        if self.training_monitor is not None:
            status = {
                'convergence_score': self.training_monitor.get_convergence_score(),
                'best_loss': self.training_monitor.best_loss,
                'plateau_detected': self.training_monitor.plateau_detected,
                'patience_counter': self.training_monitor.patience_counter
            }
            return status
        return {}
    
    def predict_sequence(self, seed: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Generate a sequence by predicting future values.
        
        Args:
            seed: Initial input values
            n_steps: Number of steps to predict
            
        Returns:
            Predicted sequence
        """
        self.training_mode = False  # Disable dropout
        h = np.zeros((self.hidden_size, 1))
        predictions = []
        
        # Warm up with seed
        for i in range(len(seed)):
            x = seed[i].reshape(-1, 1)
            h_raw = np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh
            h = self.activation(h_raw)
            y = np.dot(self.Why, h) + self.by
            predictions.append(y.flatten())
        
        # Generate future predictions
        last_input = predictions[-1] if predictions else seed[-1]
        for _ in range(n_steps):
            x = last_input.reshape(-1, 1)
            h_raw = np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh
            h = self.activation(h_raw)
            y = np.dot(self.Why, h) + self.by
            predictions.append(y.flatten())
            last_input = y.flatten()
        
        return np.array(predictions)
    
    def save_model(self, filepath: str) -> None:
        """Save model to file (supports multi-layer)."""
        model_data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'sequence_length': self.sequence_length,
            'activation_type': self.activation_type,
            'dropout_rate': self.dropout_rate,
            'optimizer_type': self.optimizer_type,
            'lr_schedule_type': self.lr_schedule_type,
            'num_layers': self.num_layers,
            'hidden_sizes': self.hidden_sizes,
            'Wxh_layers': self.Wxh_layers,
            'Whh_layers': self.Whh_layers,
            'bh_layers': self.bh_layers,
            'Why': self.Why,
            'by': self.by,
            'loss_history': self.loss_history,
            'epoch_losses': self.epoch_losses
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @staticmethod
    def load_model(filepath: str) -> 'RNNModel':
        """Load model from file (supports multi-layer)."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle backward compatibility
        dropout_rate = model_data.get('dropout_rate', 0.0)
        optimizer_type = model_data.get('optimizer_type', 'sgd')
        lr_schedule_type = model_data.get('lr_schedule_type', 'constant')
        num_layers = model_data.get('num_layers', 1)
        hidden_sizes = model_data.get('hidden_sizes', None)
        
        model = RNNModel(
            input_size=model_data['input_size'],
            hidden_size=model_data['hidden_size'],
            output_size=model_data['output_size'],
            learning_rate=model_data['learning_rate'],
            sequence_length=model_data['sequence_length'],
            activation=model_data['activation_type'],
            dropout_rate=dropout_rate,
            optimizer_type=optimizer_type,
            lr_schedule=lr_schedule_type,
            num_layers=num_layers,
            hidden_sizes=hidden_sizes
        )
        
        # Load weights
        if 'Wxh_layers' in model_data:
            # Multi-layer model
            model.Wxh_layers = model_data['Wxh_layers']
            model.Whh_layers = model_data['Whh_layers']
            model.bh_layers = model_data['bh_layers']
            # Update backward compatibility variables
            model.Wxh = model.Wxh_layers[0]
            model.Whh = model.Whh_layers[0]
            model.bh = model.bh_layers[0]
        else:
            # Old single-layer model
            model.Wxh_layers[0] = model_data['Wxh']
            model.Whh_layers[0] = model_data['Whh']
            model.bh_layers[0] = model_data['bh']
            model.Wxh = model_data['Wxh']
            model.Whh = model_data['Whh']
            model.bh = model_data['bh']
        
        model.Why = model_data['Why']
        model.by = model_data['by']
        model.loss_history = model_data['loss_history']
        model.epoch_losses = model_data['epoch_losses']
        
        return model
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters as dictionary."""
        # Calculate total parameters for multi-layer model
        total_params = self.Why.size + self.by.size
        for layer in range(self.num_layers):
            total_params += self.Wxh_layers[layer].size
            total_params += self.Whh_layers[layer].size
            total_params += self.bh_layers[layer].size
        
        params = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'sequence_length': self.sequence_length,
            'activation': self.activation_type,
            'dropout_rate': self.dropout_rate,
            'optimizer_type': self.optimizer_type,
            'lr_schedule': self.lr_schedule_type,
            'num_layers': self.num_layers,
            'hidden_sizes': self.hidden_sizes,
            'total_parameters': total_params
        }
        
        # Add current learning rate if scheduler is active
        if self.lr_scheduler is not None:
            params['current_lr'] = self.lr_scheduler.get_lr()
        
        return params
