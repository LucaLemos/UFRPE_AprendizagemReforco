from .qtable_helper import epsilon_greedy, epsilon_greedy_probs
import numpy as np
from collections import deque

# Algoritmo Expected-SARSA
def run_expected_sarsa(env, steps, lr=0.1, gamma=0.95, epsilon=0.1, verbose=True):
    print(f"Run Expected_sar in {env}")
    
    num_actions = env.action_space.n

    # inicializa a tabela Q toda com zeros
    # usar o estado como índice das linhas e a ação como índice das colunas
    Q = np.zeros(shape = (env.observation_space.n, num_actions))

    # para cada episódio, guarda sua soma de recompensas (retorno não-descontado)
    sum_rewards_per_ep = []
    sum_rewards, reward = 0, 0
    state, _ = env.reset()
    done = False
    count_ep = 0

    replay_buffer = deque(maxlen=steps)
    # loop principal
    for i in range(steps):
        # escolhe a próxima ação -- usa epsilon-greedy
        action = epsilon_greedy(Q, state, epsilon)
        # realiza a ação, ou seja, dá um passo no ambiente
        next_state, reward, terminated, trunc, info = env.step(action)
        done = terminated or trunc
        if terminated:
            # para estados terminais
            V_next_state = 0
        else:
            # para estados não-terminais -- valor esperado
            p_next_actions = epsilon_greedy_probs(Q, next_state, num_actions, epsilon)
            V_next_state = np.sum( np.asarray(p_next_actions) * Q[next_state] )
        # atualiza a Q-table
        # delta = (estimativa usando a nova recompensa) - estimativa antiga
        delta = (reward + gamma * V_next_state) - Q[state,action]
        Q[state,action] = Q[state,action] + lr * delta
        replay_buffer.append((state, action, reward, next_state, terminated, trunc, done, info))

        # atualiza o estado
        sum_rewards += reward
        state = next_state
        if done or i == steps - 1:
            # salva o retorno do episódio que encerrou
            sum_rewards_per_ep.append(sum_rewards)
            if verbose and count_ep % 100 == 0:
                print(f"Episódio {count_ep} terminou com recompensa {sum_rewards} na transição {i}")
            # reseta o ambiente para o novo episódio
            sum_rewards, reward = 0, 0
            state, _ = env.reset()
            done = False
            count_ep += 1
    return sum_rewards_per_ep, Q, replay_buffer

# Algoritmo SARSA
# Atenção: os espaços de estados e de ações precisam ser discretos, dados por valores inteiros
def run_sarsa(env, replay_buffer, steps, lr=0.1, gamma=0.95, epsilon=0.1, verbose=True):
    print(f"Run sarsa in {env}")

    # inicializa a tabela Q com valores aleatórios de -1.0 a 0.0
    # usar o estado como índice das linhas e a ação como índice das colunas
    Q = np.zeros(shape = (env.observation_space.n, env.action_space.n))

    # para cada episódio, guarda sua soma de recompensas (retorno não-descontado)
    sum_rewards_per_ep = []
    sum_rewards, reward = 0, 0
    state, _ = env.reset()
    done = False
    count_ep = 0

    print("OI")
    # loop principal
    action = epsilon_greedy(Q, state, epsilon)
    for i in range(steps):
        # realiza a ação, ou seja, dá um passo no ambiente
        next_state, reward, terminated, trunc, _ = env.step(action)
        done = terminated or trunc
        next_action = epsilon_greedy(Q, next_state, epsilon)
        
        if terminated:
                # para estados terminais
                V_next_state = 0
        else:
                # para estados não-terminais
                V_next_state = Q[next_state, next_action]
        # atualiza a Q-table
        # delta = (estimativa usando a nova recompensa) - estimativa antiga
        delta = (reward + gamma * V_next_state) - Q[state,action]
        Q[state,action] = Q[state,action] + lr * delta
        replay_buffer.add(state, action, reward, next_state, done)
        
        # atualiza o estado
        sum_rewards += reward
        state = next_state
        action = next_action

        if done or i == steps - 1:
            # salva o retorno do episódio que encerrou
            sum_rewards_per_ep.append(sum_rewards)
            if verbose and count_ep % 100 == 0:
                print(f"Episódio {count_ep} terminou com recompensa {sum_rewards} na transição {i}")
            # reseta o ambiente para o novo episódio
            sum_rewards, reward = 0, 0
            state, _ = env.reset()
            action = epsilon_greedy(Q, state, epsilon)
            done = False
            count_ep += 1

    return sum_rewards_per_ep, Q