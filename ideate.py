import torch
import torch.nn as nn

batch_size = 4
k = 36
feature_dim = 2048
embedding_dim = 512
hidden_dim = 512
vocab_size = 10010

V = torch.randn(batch_size, k, feature_dim)
captions = torch.randint(0, vocab_size, (batch_size, 15))

class AttentionLayer(nn.Module):
    def __init__(self, feature_dim, embedding_dim, attention_dim):
        super.__init__()
        # Replicating the formula given in #3
        self.W_va= nn.Linear(feature_dim, attention_dim)
        self.W_ha = nn.Linear(hidden_dim, attention_dim)
        self.w_a = nn.Linear(attention_dim, 1)

    def forward(self, V, captions):
        # V = global image features. Shape of (batch, k, feature_dim)
        # h1 = (batch, hidden_dim)
        attention = torch.tanh(self.W_va(V) + self.W_ha(h1).unsqueeze(1)) # shape of (batch, k, attention_dim)
        scores = self.w_a(attention).squeeze(2) # shape of (batch, k)
        alpha = torch.softmax(scores, dim=1) # shape of (batch, k)
        v_t_hat = (alpha.unqueeze(2) * V).sum(dim=1)
        return v_t_hat, alpha
