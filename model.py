import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import SAMPLE_RATE, D_MODEL, NUM_HEADS, FF_DIM, NUM_LAYERS, DROPOUT, NUM_CLASSES

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

class CTENNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolutional Feature Extractor (as per CTENN paper)
        self.conv_blocks = nn.ModuleList([
            # Block 1
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            ),
            # Block 2
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ),
            # Block 3
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ),
            # Block 4
            nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
        ])
        
        # Feature projection to transformer dimension
        self.feature_projection = nn.Sequential(
            nn.Linear(512, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.Dropout(DROPOUT)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model=D_MODEL, max_len=5000)
        self.dropout = nn.Dropout(DROPOUT)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation='relu',
            batch_first=True,
            norm_first=False  # Post-norm as in original transformer
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=NUM_LAYERS
        )
        
        # Classification head with attention pooling
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=D_MODEL,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            batch_first=True
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, NUM_CLASSES)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following the paper's methodology"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # x: [batch_size, 1, signal_length]
        batch_size = x.size(0)
        
        # Convolutional feature extraction
        features = x
        for conv_block in self.conv_blocks:
            features = conv_block(features)
        
        # features: [batch_size, 512, reduced_length]
        features = features.permute(0, 2, 1)  # [batch_size, seq_len, 512]
        
        # Project to transformer dimension
        features = self.feature_projection(features)  # [batch_size, seq_len, D_MODEL]
        
        # Add positional encoding
        features = self.pos_encoding(features)
        features = self.dropout(features)
        
        # Transformer encoding
        encoded = self.transformer_encoder(features)  # [batch_size, seq_len, D_MODEL]
        
        # Attention pooling (instead of simple mean pooling)
        # Use the mean as query for attention pooling
        query = encoded.mean(dim=1, keepdim=True)  # [batch_size, 1, D_MODEL]
        pooled_output, _ = self.attention_pooling(query, encoded, encoded)
        pooled_output = pooled_output.squeeze(1)  # [batch_size, D_MODEL]
        
        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, NUM_CLASSES]
        
        return logits