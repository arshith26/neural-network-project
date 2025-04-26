# skillmap/deep_matcher_model.py

import torch
import torch.nn as nn

class JobDescriptionModel(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 num_layers: int = 1,
                 bidirectional: bool = True,
                 dropout: float = 0.3):
        """
        Simple LSTM-based classifier.
        
        Args:
            vocab_size:      size of the tokenizer vocabulary (+1 for OOV).
            embedding_dim:   dimensionality of word embeddings.
            hidden_dim:      hidden size of the LSTM.
            num_classes:     number of output classes (job titles).
            num_layers:      number of LSTM layers.
            bidirectional:   use bidirectional LSTM if True.
            dropout:         dropout between LSTM layers.
        """
        super(JobDescriptionModel, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=0)
        
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        
        # If bidirectional, hidden_dim * 2; else hidden_dim
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_length) with token indices.
        Returns:
            logits: Tensor of shape (batch_size, num_classes).
        """
        # 1) Embedding lookup => (batch, seq_len, embedding_dim)
        emb = self.embedding(input_ids)
        
        # 2) LSTM => output (batch, seq_len, hidden_dim * num_directions)
        lstm_out, _ = self.lstm(emb)
        
        # 3) Pooling: take the last timestep for classification
        #    Alternatively, you could do mean or max pooling across seq_len.
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim * num_directions)
        
        # 4) Feed through fully-connected layers
        logits = self.fc(last_hidden)     # (batch, num_classes)
        return logits
