import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
emoji = "😊"
tokens = tokenizer.encode(emoji)
print("Tokens:", tokens)
for t in tokens:
    try:
        print(f"Token {t} decoded as string:", tokenizer.decode([t]))
    except Exception as e:
        print(f"Token {t} decode error:", e)
    print(f"Token {t} decoded as bytes:", tokenizer.decode_single_token_bytes(t))

print("Decoded together:", tokenizer.decode(tokens))
