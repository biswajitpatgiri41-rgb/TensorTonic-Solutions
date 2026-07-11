def validate_records(records, schema):
    type_map = {
        "int": int,
        "float": float,
        "str": str,
    }

    results = []

    for i, record in enumerate(records):
        errors = []

        for field in schema:
            column = field["column"]

            if column not in record:
                errors.append(f"{column}: missing")
                continue

            value = record[column]

            if value is None:
                if not field["nullable"]:
                    errors.append(f"{column}: null")
                continue

            expected = field["type"]

            if expected == "float":
                if type(value) not in (int, float):
                    errors.append(f"{column}: expected float, got {type(value).__name__}")
                    continue
            else:
                if type(value) is not type_map[expected]:
                    errors.append(f"{column}: expected {expected}, got {type(value).__name__}")
                    continue

            if expected in ("int", "float"):
                if "min" in field and value < field["min"]:
                    errors.append(f"{column}: out of range")
                    continue
                if "max" in field and value > field["max"]:
                    errors.append(f"{column}: out of range")

        results.append((i, len(errors) == 0, errors))

    return results