from base import Tokenizer
import json

with open("../dataset/dataset.txt") as file:
    text = file.read()

tokenizer = Tokenizer()
tokenizer.fit(text)


encoded = tokenizer.encode('hello')
print("Encoded:", encoded)

vocab_size = len(tokenizer.word_to_id)

print(vocab_size)

tokenizer_data = {
    "word_to_id": tokenizer.word_to_id,
    "id_to_word": tokenizer.id_to_word
}

# with open("tokenizer.json", "w") as file:
#     json.dump(tokenizer_data, file, indent=4)


