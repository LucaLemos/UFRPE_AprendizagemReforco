from .qtable_helper import epsilon_greedy, epsilon_greedy_probs
import numpy as np
from tianshou.data import ReplayBuffer, Batch

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
    
    # ReplayBuffer para armazenar as transições
    replay_buffer = ReplayBuffer(size=steps)

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

        replay_buffer.add(
            Batch(
                obs=state,
                act=action,
                rew=reward,
                obs_next=next_state,
                terminated=terminated,
                truncated=trunc,
                done=done,
                info=info
            )
        )
        
        sum_rewards += reward
        state = next_state
        if done or i == steps - 1:
            # salva o retorno do episódio que encerrou
            sum_rewards_per_ep.append(sum_rewards)
            if verbose and count_ep % 100 == 0:
                print(f"Episódio {count_ep} {"NÃO terminou e teve" if i == steps - 1 else "terminou com"} recompensa {sum_rewards} na transição {i}")
            # reseta o ambiente para o novo episódio
            sum_rewards, reward = 0, 0
            state, _ = env.reset()
            done = False
            count_ep += 1
    return sum_rewards_per_ep, Q, replay_buffer