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
        # basic LSTM classifier setup
        super(JobDescriptionModel, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input: (batch_size, seq_length)
        x = self.embedding(input_ids)

        # run through LSTM
        lstm_out, _ = self.lstm(x)

        # take last time step
        last_hidden = lstm_out[:, -1, :]

        # pass through dense layers
        logits = self.fc(last_hidden)
        return logits
