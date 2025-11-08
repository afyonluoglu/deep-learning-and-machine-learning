"""
Self-Attention Modülü
Self-Attention mekanizmasının implementasyonu ve eğitim fonksiyonları
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Callable, Optional


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention katmanı"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        print(f"🟢 MultiHeadSelfAttention: d_model={d_model}, num_heads={num_heads}")
        assert d_model % num_heads == 0, f"d_model, num_heads'e tam bölünmelidir! ({d_model/num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Her head için boyut
        
        # Query, Key, Value projeksiyon matrisleri
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projeksiyon
        self.W_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Attention ağırlıklarını sakla
        self.last_attention_weights = None
        self.last_qkv = None
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) - opsiyonel
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projeksiyonlar
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # QKV'yi sakla (görselleştirme için)
        self.last_qkv = {
            'Q': Q.detach().cpu().numpy(),
            'K': K.detach().cpu().numpy(),
            'V': V.detach().cpu().numpy()
        }
        
        # Multi-head için reshape
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        
        # QK^T hesapla
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        # (batch_size, num_heads, seq_len, seq_len)
        
        # Mask uygula (varsa)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax uygula
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Attention weights'i sakla (görselleştirme için)
        self.last_attention_weights = attention_weights.detach().cpu().numpy()
        
        # Attention uygula
        attended_values = torch.matmul(attention_weights, V)
        # (batch_size, num_heads, seq_len, d_k)
        
        # Heads'leri birleştir
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.W_o(attended_values)
        
        return output
    
    def get_attention_weights(self):
        """Son attention ağırlıklarını döndür"""
        return self.last_attention_weights
    
    def get_qkv_matrices(self):
        """Son QKV matrislerini döndür"""
        return self.last_qkv


class TransformerBlock(nn.Module):
    """Transformer bloğu (Self-Attention + Feed Forward)"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Multi-Head Self-Attention
        print("🚩 self.attention is initialized")
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-Attention with residual connection
        attended = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed Forward with residual connection
        fed_forward = self.ffn(x)
        x = self.norm2(x + self.dropout(fed_forward))
        
        return x


class SelfAttentionModel(nn.Module):
    """Self-Attention tabanlı basit model"""
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        print("🟡 SelfAttentionModel: Embedding layer initialized")
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)        
        print("🟡 SelfAttentionModel: PositionalEncoding layer initialized")
        
        # Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        print(f"🟡 SelfAttentionModel: {num_layers} Transformer blocks initialized")
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        print("🟡 SelfAttentionModel: Output layer initialized")
        
    def forward(self, x, mask=None):
        # Embedding
        x = self.embedding(x) * np.sqrt(self.d_model)
        
        # Positional Encoding
        x = self.pos_encoding(x)
        
        # Transformer Blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        # Output
        output = self.fc_out(x)
        
        return output
    
    def get_attention_weights(self):
        """Tüm katmanların attention ağırlıklarını döndür"""
        weights = []
        for layer in self.layers:
            w = layer.attention.get_attention_weights()
            if w is not None:
                weights.append(w)
        return weights
    
    def get_qkv_matrices(self):
        """İlk katmanın QKV matrislerini döndür"""
        return self.layers[0].attention.get_qkv_matrices()


class PositionalEncoding(nn.Module):
    """Positional Encoding"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Positional encoding matrisini oluştur
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SelfAttentionTrainer:
    """Self-Attention modeli için eğitim sınıfı"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Varsayılan parametreler
        self.d_model = 64
        self.num_heads = 4
        self.num_layers = 2
        self.dropout = 0.1
        self.learning_rate = 0.001
        
        # Model ve diğer değişkenler
        self.model = None
        self.vocab = None
        self.token_to_idx = None
        self.idx_to_token = None
        self.history = {'loss': [], 'epoch': []}
        
    def set_d_model(self, d_model: int):
        """d_model parametresini ayarla"""
        self.d_model = d_model
        # print(f"🔧 d_model set to {self.d_model}")
        
    def set_num_heads(self, num_heads: int):
        """num_heads parametresini ayarla"""
        self.num_heads = num_heads
        
    def set_dropout(self, dropout: float):
        """dropout parametresini ayarla"""
        self.dropout = dropout
        
    def set_learning_rate(self, lr: float):
        """learning rate parametresini ayarla"""
        self.learning_rate = lr
        
    def build_vocab(self, tokens: List[str]):
        """Vocabulary oluştur"""
        self.vocab = list(set(tokens))
        self.vocab.sort()
        
        # Özel tokenlar ekle
        self.vocab = ['<PAD>', '<UNK>'] + self.vocab
        
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        
    def tokenize(self, tokens: List[str]) -> torch.Tensor:
        """Token'ları index'lere çevir"""
        indices = [self.token_to_idx.get(token, self.token_to_idx['<UNK>']) 
                   for token in tokens]
        return torch.tensor(indices, dtype=torch.long)
    
    def create_training_data(self, tokens: List[str], batch_size: int):
        """Eğitim verisi oluştur (basit next-token prediction)"""
        # Token'ları index'lere çevir
        indices = self.tokenize(tokens).unsqueeze(0)  # (1, seq_len)
        
        # Input: tüm tokenlar (son hariç)
        # Target: tüm tokenlar (ilk hariç)
        X = indices[:, :-1]
        y = indices[:, 1:]
        
        # Dataset oluştur
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    def train(self, tokens: List[str], epochs: int = 50, batch_size: int = 8,
              progress_callback: Optional[Callable] = None) -> Dict:
        """Modeli eğit"""
        
        # Vocabulary oluştur
        self.build_vocab(tokens)
        vocab_size = len(self.vocab)
        
        print("Creating model...")
        # Model oluştur
        self.model = SelfAttentionModel(
            vocab_size=vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss ve optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Eğitim verisi
        dataloader = self.create_training_data(tokens, batch_size)
        
        # Eğitim döngüsü
        self.history = {'loss': [], 'epoch': []}
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                output = self.model(batch_X)  # (batch, seq_len, vocab_size)
                
                # Loss hesapla
                output = output.view(-1, vocab_size)
                batch_y = batch_y.view(-1)
                loss = criterion(output, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.history['loss'].append(avg_loss)
            self.history['epoch'].append(epoch)
            
            # Progress callback
            if progress_callback:
                progress_callback(epoch, epochs, avg_loss)
        
        return self.history
    
    def get_attention_weights(self, tokens: List[str]) -> np.ndarray:
        """Verilen tokenlar için attention weights hesapla"""
        if self.model is None:
            return None
        
        self.model.eval()
        
        with torch.no_grad():
            # Token'ları hazırla
            indices = self.tokenize(tokens).unsqueeze(0).to(self.device)
            
            # Forward pass
            _ = self.model(indices)
            
            # Attention weights'i al
            weights = self.model.get_attention_weights()
            
            if weights:
                # İlk batch, ilk head'i al
                return weights[0][0, 0, :, :]  # (seq_len, seq_len)
            
        return None
    
    def get_qkv_matrices(self, tokens: List[str]) -> Dict:
        """QKV matrislerini al"""
        if self.model is None:
            return None
        
        self.model.eval()
        
        with torch.no_grad():
            # Token'ları hazırla
            indices = self.tokenize(tokens).unsqueeze(0).to(self.device)
            
            # Forward pass
            _ = self.model(indices)
            
            # QKV matrislerini al
            qkv = self.model.get_qkv_matrices()
            
            if qkv:
                # İlk batch'i al
                return {
                    'Q': qkv['Q'][0],  # (seq_len, d_model)
                    'K': qkv['K'][0],
                    'V': qkv['V'][0]
                }
        
        return None
    
    def get_training_history(self) -> Dict:
        """Eğitim geçmişini döndür"""
        return self.history
    
    def get_config(self) -> Dict:
        """Model konfigürasyonunu döndür"""
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'vocab': self.vocab,
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token
        }
    
    def load_model(self, model: nn.Module, config: Dict):
        """Model ve konfigürasyonu yükle"""
        self.model = model.to(self.device)
        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.learning_rate = config['learning_rate']
        self.vocab = config['vocab']
        self.token_to_idx = config['token_to_idx']
        self.idx_to_token = config['idx_to_token']
