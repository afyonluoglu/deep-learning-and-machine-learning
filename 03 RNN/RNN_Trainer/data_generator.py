"""
Time Series Data Generator for RNN Training
"""
import numpy as np
from typing import Tuple


class DataGenerator:
    """Generate various time series patterns for RNN training."""
    
    @staticmethod
    def generate_sine_wave(n_samples: int = 1000, 
                          frequency: float = 1.0,
                          amplitude: float = 1.0,
                          noise_level: float = 0.0,
                          phase: float = 0.0) -> np.ndarray:
        """
        Generate sine wave data.
        
        Args:
            n_samples: Number of samples
            frequency: Frequency of the sine wave
            amplitude: Amplitude of the sine wave
            noise_level: Standard deviation of Gaussian noise
            phase: Phase shift
            
        Returns:
            Sine wave data
        """
        t = np.linspace(0, 4 * np.pi * frequency, n_samples)
        data = amplitude * np.sin(t + phase)
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, n_samples)
            data += noise
        
        return data.reshape(-1, 1)
    
    @staticmethod
    def generate_cosine_wave(n_samples: int = 1000,
                            frequency: float = 1.0,
                            amplitude: float = 1.0,
                            noise_level: float = 0.0,
                            phase: float = 0.0) -> np.ndarray:
        """Generate cosine wave data."""
        t = np.linspace(0, 4 * np.pi * frequency, n_samples)
        data = amplitude * np.cos(t + phase)
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, n_samples)
            data += noise
        
        return data.reshape(-1, 1)
    
    @staticmethod
    def generate_square_wave(n_samples: int = 1000,
                            frequency: float = 1.0,
                            amplitude: float = 1.0,
                            noise_level: float = 0.0) -> np.ndarray:
        """Generate square wave data."""
        t = np.linspace(0, 4 * np.pi * frequency, n_samples)
        data = amplitude * np.sign(np.sin(t))
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, n_samples)
            data += noise
        
        return data.reshape(-1, 1)
    
    @staticmethod
    def generate_sawtooth_wave(n_samples: int = 1000,
                              frequency: float = 1.0,
                              amplitude: float = 1.0,
                              noise_level: float = 0.0) -> np.ndarray:
        """Generate sawtooth wave data."""
        t = np.linspace(0, 4 * np.pi * frequency, n_samples)
        data = amplitude * (2 * (t * frequency / (2 * np.pi) - 
                                 np.floor(t * frequency / (2 * np.pi) + 0.5)))
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, n_samples)
            data += noise
        
        return data.reshape(-1, 1)
    
    @staticmethod
    def generate_triangular_wave(n_samples: int = 1000,
                                frequency: float = 1.0,
                                amplitude: float = 1.0,
                                noise_level: float = 0.0) -> np.ndarray:
        """Generate triangular wave data."""
        t = np.linspace(0, 4 * np.pi * frequency, n_samples)
        data = amplitude * (2 * np.abs(2 * (t * frequency / (2 * np.pi) - 
                                            np.floor(t * frequency / (2 * np.pi) + 0.5))) - 1)
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, n_samples)
            data += noise
        
        return data.reshape(-1, 1)
    
    @staticmethod
    def generate_mixed_waves(n_samples: int = 1000,
                           frequencies: list = [1.0, 2.0],
                           amplitudes: list = [1.0, 0.5],
                           noise_level: float = 0.0) -> np.ndarray:
        """
        Generate mixed sine waves.
        
        Args:
            n_samples: Number of samples
            frequencies: List of frequencies
            amplitudes: List of amplitudes
            noise_level: Noise level
            
        Returns:
            Mixed wave data
        """
        t = np.linspace(0, 4 * np.pi, n_samples)
        data = np.zeros(n_samples)
        
        for freq, amp in zip(frequencies, amplitudes):
            data += amp * np.sin(freq * t)
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, n_samples)
            data += noise
        
        return data.reshape(-1, 1)
    
    @staticmethod
    def generate_exponential(n_samples: int = 1000,
                           growth_rate: float = 0.01,
                           noise_level: float = 0.0) -> np.ndarray:
        """Generate exponential growth/decay data."""
        t = np.linspace(0, 10, n_samples)
        data = np.exp(growth_rate * t)
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, n_samples)
            data += noise
        
        return data.reshape(-1, 1)
    
    @staticmethod
    def generate_polynomial(n_samples: int = 1000,
                          coefficients: list = [1, 0.5, 0.1],
                          noise_level: float = 0.0) -> np.ndarray:
        """
        Generate polynomial data.
        
        Args:
            n_samples: Number of samples
            coefficients: Polynomial coefficients [a, b, c, ...] for a + bx + cx^2 + ...
            noise_level: Noise level
            
        Returns:
            Polynomial data
        """
        t = np.linspace(-5, 5, n_samples)
        data = np.zeros(n_samples)
        
        for i, coef in enumerate(coefficients):
            data += coef * (t ** i)
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, n_samples)
            data += noise
        
        return data.reshape(-1, 1)
    
    @staticmethod
    def generate_random_walk(n_samples: int = 1000,
                           step_size: float = 0.1,
                           start_value: float = 0.0) -> np.ndarray:
        """Generate random walk data."""
        steps = np.random.randn(n_samples) * step_size
        data = np.cumsum(steps) + start_value
        return data.reshape(-1, 1)
    
    @staticmethod
    def generate_arma(n_samples: int = 1000,
                     ar_coeffs: list = [0.5],
                     ma_coeffs: list = [0.3],
                     noise_std: float = 0.1) -> np.ndarray:
        """
        Generate ARMA (AutoRegressive Moving Average) process.
        
        Args:
            n_samples: Number of samples
            ar_coeffs: AR coefficients
            ma_coeffs: MA coefficients
            noise_std: Standard deviation of noise
            
        Returns:
            ARMA data
        """
        p = len(ar_coeffs)
        q = len(ma_coeffs)
        
        # Generate white noise
        epsilon = np.random.normal(0, noise_std, n_samples + max(p, q))
        data = np.zeros(n_samples + max(p, q))
        
        # Generate ARMA process
        for t in range(max(p, q), n_samples + max(p, q)):
            # AR component
            ar_sum = sum(ar_coeffs[i] * data[t - i - 1] for i in range(p))
            # MA component
            ma_sum = sum(ma_coeffs[i] * epsilon[t - i - 1] for i in range(q))
            data[t] = ar_sum + ma_sum + epsilon[t]
        
        return data[max(p, q):].reshape(-1, 1)
    
    @staticmethod
    def generate_damped_oscillation(n_samples: int = 1000,
                                   frequency: float = 1.0,
                                   damping: float = 0.01,
                                   amplitude: float = 1.0,
                                   noise_level: float = 0.0) -> np.ndarray:
        """Generate damped oscillation."""
        t = np.linspace(0, 10, n_samples)
        data = amplitude * np.exp(-damping * t) * np.sin(2 * np.pi * frequency * t)
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, n_samples)
            data += noise
        
        return data.reshape(-1, 1)
    
    @staticmethod
    def create_sequences(data: np.ndarray, 
                        sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences for RNN training.
        
        Args:
            data: Time series data
            sequence_length: Length of input sequences
            
        Returns:
            X (inputs) and y (targets)
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + 1:i + sequence_length + 1])
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Normalize data to range [-1, 1].
        
        Args:
            data: Input data
            
        Returns:
            Normalized data, min value, max value
        """
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val - min_val == 0:
            return data, min_val, max_val
        
        normalized = 2 * (data - min_val) / (max_val - min_val) - 1
        return normalized, min_val, max_val
    
    @staticmethod
    def denormalize_data(data: np.ndarray, 
                        min_val: float, 
                        max_val: float) -> np.ndarray:
        """Denormalize data from range [-1, 1]."""
        return (data + 1) * (max_val - min_val) / 2 + min_val
