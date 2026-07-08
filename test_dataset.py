from torch.utils.data import DataLoader
from tokenizers.base import Tokenizer # Assuming this points to your custom base.py Tokenizer
from dataset_ import LanguageDataset

with open("dataset/dataset.txt") as file:
    text = file.read()

tokenizer = Tokenizer()
tokenizer.fit(text)
tokens = tokenizer.encode(text)

dataset = LanguageDataset(
    tokens,
    context_size=3
)


BATCH_SIZE = 4  
shuffle_data = True 

dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=shuffle_data,
    drop_last=True  
)

for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
    # print(f"Batch {batch_idx + 1}:")
    # print("X Tensors (Shape: ", x_batch.shape, "):\n", x_batch)
    # print("Y Tensors (Shape: ", y_batch.shape, "):\n", y_batch)
    # print("-" * 40)
    
    break 