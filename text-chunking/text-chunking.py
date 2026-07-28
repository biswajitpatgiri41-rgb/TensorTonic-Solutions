def text_chunking(tokens, chunk_size, overlap):
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(tokens), step):
        chunks.append(tokens[i:i + chunk_size])
        if i + chunk_size >= len(tokens):
            break
    return chunks