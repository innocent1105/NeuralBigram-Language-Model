import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tokenizers.base import Tokenizer          # Your custom tokenizer[cite: 6]
from dataset_ import LanguageDataset # Your sliding window dataset[cite: 3]
from model import NeuralBigramModel

# --- 1. Setup Data Pipeline ---
with open("dataset/dataset.txt") as file:
    text = file.read()

tokenizer = Tokenizer()
tokenizer.fit(text)
tokens = tokenizer.encode(text)

# Force context_size to 1 for a true structural Bigram[cite: 4]
dataset = LanguageDataset(tokens, context_size=1)

BATCH_SIZE = 32
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# --- 2. Initialize Hyperparameters & Model ---
VOCAB_SIZE = len(tokenizer.word_to_id)
EMBEDDING_DIM = 64  # Size of the hidden word vectors

model = NeuralBigramModel(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)

# Loss function handles raw logits directly, applying LogSoftmax internally
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


EPOCHS = 300

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.long()
        y_batch = y_batch.long()
        
        # Forward pass
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
   
    if (epoch + 1) % 10 == 0 or epoch == 0:
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1:2d}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")






def generate_text(model, tokenizer, start_word="i", max_len=15):
    model.eval()  # Set model to evaluation mode
    current_word = start_word.lower()
    sentence = [current_word]
    
    word_to_id = tokenizer.word_to_id
    id_to_word = tokenizer.id_to_word

    print(f"\n Generating text starting with '{start_word}':")
    
    with torch.no_grad():
        for _ in range(max_len):
            word_id = word_to_id.get(current_word, word_to_id["<UNK>"])
            
            x = torch.tensor([[word_id]], dtype=torch.long)
            
            logits = model(x)  # Shape: (1, vocab_size)
            
            probs = torch.softmax(logits, dim=-1)
            
            next_word_id = torch.multinomial(probs, num_samples=1).item()
            
            next_word = id_to_word.get(next_word_id, "<UNK>")
            
            if next_word == "<EOS>":
                break
                
            sentence.append(next_word)
            current_word = next_word  
            
    return " ".join(sentence)


generated_phrase = generate_text(model, tokenizer, start_word="awesome ", max_len=10)
print(generated_phrase)