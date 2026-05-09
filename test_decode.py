import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
tokens = tokenizer.encode("😊")
byte_buf = bytearray()
for t in tokens:
    b = tokenizer.decode_single_token_bytes(t)
    byte_buf.extend(b)
    try:
        text = byte_buf.decode("utf-8")
        print(f"Decoded successfully: {text}")
        byte_buf.clear()
    except UnicodeDecodeError:
        print(f"Incomplete sequence for token {t}")
