from performer_pytorch import Performer

class PerformerModel:
    def __init__(self, num_tokens, dim, depth, heads, causal=False):
        self.model = Performer(
            dim=dim,
            depth=depth,
            heads=heads,
            causal=causal,
            num_tokens=num_tokens
        )

    def forward(self, x):
        return self.model(x)

