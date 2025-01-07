from linformer import Linformer
import torch.nn as nn

class LinformerModel(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, k):
        super(LinformerModel, self).__init__()
        self.model = Linformer(
            dim=dim,
            seq_len=seq_len,
            depth=depth,
            heads=heads,
            k=k
        )

    def forward(self, x):
        return self.model(x)
