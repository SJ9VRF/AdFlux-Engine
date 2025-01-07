from transformers import GPT2Model, GPT2Config
import torch.nn as nn

class DecisionTransformer(nn.Module):
    """
    Decision Transformer for RL as sequence modeling.
    """
    def __init__(self, state_dim, action_dim, max_length, hidden_dim=128):
        super(DecisionTransformer, self).__init__()
        config = GPT2Config(
            n_embd=hidden_dim,
            n_layer=6,
            n_head=8,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            vocab_size=1
        )
        self.transformer = GPT2Model(config)
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.return_embed = nn.Linear(1, hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, states, actions, returns):
        """
        Forward pass through the Decision Transformer.
        """
        state_embeddings = self.state_embed(states)
        action_embeddings = self.action_embed(actions)
        return_embeddings = self.return_embed(returns)

        embeddings = state_embeddings + action_embeddings + return_embeddings
        transformer_output = self.transformer(inputs_embeds=embeddings)
        logits = self.head(transformer_output.last_hidden_state)
        return logits

