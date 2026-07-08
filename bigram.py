from collections import defaultdict

class BigramModel: 
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()

    def train(self, text):
        tokens = text.lower().split()

        for token in tokens:
            self.vocab.add(token)
        
        for w1, w2 in zip(tokens[:-1], tokens[1:]):
            self.counts[w1][w2] += 1

    def predict(self, word):
        if word not in self.counts:
            return None
        
        next_words = self.counts[word]
        return max(next_words, key=next_words.get)
    
    def probabilities(self, word):
        if word not in self.counts:
            return {}
        
        total = sum(self.counts[word].values())

        return {w: c / total for w, c in self.counts[word].items()}
    


text = """
    Hello there
    hey how are you?
    how are you doing
    how is home ?
    good morning how are you?
    morning there?
"""


model = BigramModel()
model.train(text)

sentence = []
for i in range(4):
    if not sentence:
        word = "hey"
    else:
        word = model.predict(sentence[-1])
    
    if word is None:
        break
    
    sentence.append(word)
for word in sentence:
    print(word, end=' ')