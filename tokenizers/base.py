class Tokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
    def fit(self, text):

        words = text.lower().split()

        unique_words = sorted(set(words))

        special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>"
        ]

        unique_words = special_tokens + unique_words

        for i, word in enumerate(unique_words):
            self.word_to_id[word] = i
            self.id_to_word[i] = word

    def encode(self, text):

        words = text.lower().split()

        return [
            self.word_to_id.get(
                word,
                self.word_to_id["<UNK>"]
            )
            for word in words
        ]
    
    def decode(self, ids):
        words = [self.id_to_word[i] for i in ids if i in self.id_to_word]
        return ' '.join(words)
    
