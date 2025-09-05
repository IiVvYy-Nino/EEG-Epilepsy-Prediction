"""Advanced EEG Sequence classifier with attention mechanisms.

Multi-scale attention BiLSTM backbone:
- Temporal attention: Focus on seizure onset patterns
- Channel attention: Highlight pathological brain regions  
- Self-attention: Capture long-range dependencies
- Optimized for 8GB GPU memory (RTX 4060)
"""
import torch
from torch import nn
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    """Channel attention for EEG feature selection across frequency bands."""
    
    def __init__(self, in_features: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [B, T, F]
        b, t, f = x.size()
        
        # Global pooling across time dimension
        avg_out = self.fc(self.avg_pool(x.transpose(1, 2)).squeeze(-1))  # [B, F]
        max_out = self.fc(self.max_pool(x.transpose(1, 2)).squeeze(-1))  # [B, F]
        
        attention = self.sigmoid(avg_out + max_out).unsqueeze(1)  # [B, 1, F]
        return x * attention, attention.squeeze(1)


class TemporalAttention(nn.Module):
    """Temporal attention for seizure pattern recognition."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, mask=None):
        # x: [B, T, H]
        batch_size, seq_len, _ = x.size()
        
        # Multi-head attention
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # mask: [B, T] -> [B, 1, 1, T] for broadcasting
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            scores.masked_fill_(mask_expanded == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        output = self.out_linear(context)
        output = self.layer_norm(output + x)  # Residual connection
        
        return output, attention_weights.mean(dim=1)  # Average across heads


class BiLSTMWithAttention(nn.Module):
    """Enhanced BiLSTM with multi-scale attention for EEG seizure detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, 
                 dropout: float = 0.1, use_attention: bool = True, attention_heads: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Channel attention for input features
        self.channel_attention = ChannelAttention(input_dim, reduction=4)
        
        # BiLSTM backbone - 更大的hidden_dim充分利用GPU
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        
        # Temporal attention
        if use_attention:
            self.temporal_attention = TemporalAttention(lstm_output_dim, num_heads=attention_heads)
            
        # Enhanced classifier with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 4, num_classes)
        )
        
        # 权重初始化
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights for numerical stability."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.normal_(param, 0, 0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
    def forward(self, x, lengths=None):
        """
        Forward pass with attention mechanisms.
        
        Args:
            x: [B, T, F] input features
            lengths: Optional[List[int]] sequence lengths for packing
            
        Returns:
            logits: [B, T, C] class predictions
            attention_info: Dict with attention weights for visualization
        """
        batch_size, seq_len, _ = x.size()
        
        # Channel attention on input features
        x_attended, channel_weights = self.channel_attention(x)
        
        # BiLSTM processing
        if lengths is not None:
            # Pack sequences for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                x_attended, lengths=lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x_attended)
        
        # Temporal attention
        attention_info = {'channel_attention': channel_weights}
        
        if self.use_attention:
            # Create padding mask for attention (1 for valid positions, 0 for padding)
            if lengths is not None:
                mask = torch.zeros(batch_size, seq_len, device=x.device, dtype=torch.bool)
                for i, length in enumerate(lengths):
                    if length < seq_len:
                        mask[i, :length] = 1
                    else:
                        mask[i, :] = 1  # All positions valid if length >= seq_len
            else:
                mask = None
                
            attended_out, temporal_weights = self.temporal_attention(lstm_out, mask)
            attention_info['temporal_attention'] = temporal_weights
            final_features = attended_out
        else:
            final_features = lstm_out
        
        # Classification
        logits = self.classifier(final_features)
        
        return logits, attention_info
    
    def get_model_size(self):
        """Calculate model parameters and memory usage."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory (weights + gradients + optimizer states)
        param_memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        training_memory_mb = param_memory_mb * 3  # weights + gradients + Adam states
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'param_memory_mb': param_memory_mb,
            'training_memory_mb': training_memory_mb
        }


# 向后兼容性别名
BiLSTMClassifier = BiLSTMWithAttention

