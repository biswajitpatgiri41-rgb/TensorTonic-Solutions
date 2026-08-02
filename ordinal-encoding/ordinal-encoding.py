def ordinal_encoding(values, ordering):
    mapping = {v: i for i, v in enumerate(ordering)}
    return [mapping[v] for v in values]