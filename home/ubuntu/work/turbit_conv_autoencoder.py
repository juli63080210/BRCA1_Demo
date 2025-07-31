import torch.nn as nn

class TurbitConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=5, padding=2), # (N, 16, L)
            nn.ReLU(),
            nn.MaxPool1d(2),                           # (N, 16, L/2)
            nn.Conv1d(16, 32, kernel_size=3, padding=1), # (N, 32, L/2)
            nn.ReLU(),
            nn.MaxPool1d(2)                            # (N, 32, L/4)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # (N, 16, L/2)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 4, kernel_size=5, stride=2, padding=2, output_padding=1),  # (N, 4, L)
            nn.Sigmoid() # Output probabilities for one-hot encoding
        )

    def forward(self, x):
        # x is (N, L, 4), permute to (N, 4, L) for Conv1d
        x = x.permute(0, 2, 1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Permute back to (N, L, 4) for consistency with input
        return decoded.permute(0, 2, 1)


