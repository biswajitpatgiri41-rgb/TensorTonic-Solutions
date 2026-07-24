def feature_store_lookup(feature_store, requests, defaults):
    result = []
    for request in requests:
        offline = feature_store.get(request["user_id"], defaults)
        result.append({**offline, **request["online_features"]})
    return result