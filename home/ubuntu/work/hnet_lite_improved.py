import torch
import torch.nn as nn

class HNetLiteImproved(nn.Module):
    def __init__(self):
        super().__init__()
        self.reductor = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Output shape (N, 32, 1)
        )
        
        # Simple attention mechanism (e.g., Squeeze-and-Excitation like)
        self.attention_fc1 = nn.Linear(32, 16)
        self.attention_relu = nn.ReLU()
        self.attention_fc2 = nn.Linear(16, 32)
        self.attention_sigmoid = nn.Sigmoid()

        self.classificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x is (N, L, 4), permute to (N, 4, L) for Conv1d
        x = x.permute(0, 2, 1)
        
        # Feature extraction
        features = self.reductor(x) # Output (N, 32, 1)
        
        # Apply attention
        # Squeeze: (N, 32, 1) -> (N, 32)
        squeezed = features.squeeze(2)
        
        # Excitation
        excitation = self.attention_fc1(squeezed)
        excitation = self.attention_relu(excitation)
        excitation = self.attention_fc2(excitation)
        excitation = self.attention_sigmoid(excitation)
        
        # Scale features with attention weights
        attended_features = features * excitation.unsqueeze(2) # (N, 32, 1) * (N, 32, 1)

        x = self.classificador(attended_features)
        return x


