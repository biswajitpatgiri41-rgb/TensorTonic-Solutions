def deduplicate(records, key_columns, strategy):
    order = []
    selected = {}

    for record in records:
        key = tuple(record.get(col) for col in key_columns)
        if key not in selected:
            order.append(key)
            selected[key] = record
        else:
            if strategy == "last":
                selected[key] = record
            elif strategy == "most_complete":
                current = selected[key]
                current_none = sum(v is None for v in current.values())
                new_none = sum(v is None for v in record.values())
                if new_none < current_none:
                    selected[key] = record

    return [selected[key] for key in order]