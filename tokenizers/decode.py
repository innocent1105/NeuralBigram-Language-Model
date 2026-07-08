from base import Tokenizer
with open("../dataset/dataset.txt") as file:
    text = file.read()


tokenizer = Tokenizer()
tokenizer.fit(text)

encoded = tokenizer.encode("hello")
print("Encoded:", encoded)

decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)