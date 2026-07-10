def value_iteration_step(values, transitions, rewards, gamma):
    new_values = []
    n_states = len(values)
    for s in range(n_states):
        best = float("-inf")
        for a in range(len(transitions[s])):
            q = rewards[s][a]
            for s_next in range(n_states):
                q += gamma * transitions[s][a][s_next] * values[s_next]
            if q > best:
                best = q
        new_values.append(float(best))
    return new_values