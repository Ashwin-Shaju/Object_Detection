from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
print(tokenizer.encode("Hello, how are you?").tokens)
