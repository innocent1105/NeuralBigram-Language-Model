import torch
import torch.nn as nn

class NeuralBigramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # Maps token IDs to continuous dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Projects embedding vector back to vocabulary space for predictions
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, context_size) -> e.g., (32, 1)
        
        # 1. Pass through embedding layer
        # out shape: (batch_size, context_size, embedding_dim)
        out = self.embedding(x)
        
        # 2. Squeeze out the context dimension since context_size = 1
        # out shape becomes: (batch_size, embedding_dim)
        out = out.squeeze(1)
        
        # 3. Project to raw logits for every word in the vocabulary
        # logits shape: (batch_size, vocab_size)
        logits = self.linear(out)
        
        return logits