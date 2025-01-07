from reformer_pytorch import Reformer

class ReformerModel:
    def __init__(self, dim, depth, max_seq_len, heads, lsh_dropout, causal=False):
        self.model = Reformer(
            dim=dim,
            depth=depth,
            max_seq_len=max_seq_len,
            heads=heads,
            lsh_dropout=lsh_dropout,
            causal=causal
        )

    def forward(self, x):
        return self.model(x)
