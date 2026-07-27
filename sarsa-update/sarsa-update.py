def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    new_q = [row[:] for row in q_table]
    td = reward + gamma * q_table[next_state][next_action] - q_table[state][action]
    new_q[state][action] += alpha * td
    return new_q