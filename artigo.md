# Offline Reinforcement Learning: Aprendizado a Partir de Experiências Pré-Coletadas

## Introdução

O aprendizado por reforço (RL) tradicional envolve a interação contínua de um agente com o ambiente, permitindo que ele colete experiências e ajuste sua política com base no feedback recebido. No entanto, em muitos cenários práticos, essa abordagem pode ser inviável devido a restrições de tempo, custo ou segurança. É aqui que entra o **Offline Reinforcement Learning (Offline RL)**, que permite que agentes aprendam a partir de um **dataset fixo de experiências**.

Neste artigo, vamos explorar o conceito de Offline RL, suas aplicações potenciais, apresentar um dataset de exemplo e discutir um algoritmo simples para treinar um agente de RL com dados fixos. Finalizaremos com a análise dos resultados obtidos.

---

## O que é Offline Reinforcement Learning?

No **Offline RL**, o agente não coleta novas interações com o ambiente durante o treinamento. Em vez disso, ele aprende exclusivamente a partir de um **conjunto fixo de transições (Replay Buffer)** que contém estados, ações, recompensas e estados seguintes. O objetivo é encontrar uma política ótima com base nesses dados.

A principal diferença entre **Offline RL** e **RL tradicional** é que, no RL convencional, o agente pode experimentar novas ações e explorar o ambiente ativamente, enquanto no Offline RL ele precisa extrair conhecimento exclusivamente do dataset fornecido.

### Desafios do Offline RL

1. **Distribuição de Dados Limitada**: O agente está restrito às experiências registradas, podendo nunca ver ações ou estados críticos para um bom desempenho.
2. **Desvio de Política (Policy Shift)**: O agente pode tentar otimizar ações que não estão bem representadas nos dados.
3. **Aprendizado Restrito**: Sem interatividade, erros nos dados ou políticas subótimas podem afetar negativamente o treinamento.

---

## Explicação de Algoritmos

### SARSA (State-Action-Reward-State-Action)
SARSA é um método de aprendizado por reforço baseado em controle da política. O nome SARSA vem da sequência de elementos que ele usa para atualizar a função de valor Q: **(s, a, r, s', a')**, onde:
- **s**: Estado atual
- **a**: Ação tomada
- **r**: Recompensa recebida
- **s'**: Próximo estado
- **a'**: Próxima ação escolhida pela política

A atualização da função Q segue a equação:
```math
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]
```

Diferente do Q-learning, que usa a melhor ação futura para a atualização, SARSA segue a política atual para selecionar ações.

### DQN (Deep Q-Network)
DQN é um algoritmo baseado em redes neurais para aprendizado por reforço. Ele usa uma rede neural para aproximar a função Q e melhorar a tomada de decisão.

A atualização da função Q segue:
```math
Q(s, a) \leftarrow r + \gamma \max Q(s', a')
```

DQN resolve problemas comuns do aprendizado por reforço, como instabilidade e correlação entre amostras consecutivas, utilizando técnicas como **Replay Buffer** e **Redes-Alvo** para estabilizar o treinamento.

---

## Aplicações de Offline RL

Apesar dos desafios, o Offline RL tem aplicações promissoras:

- **Saúde**: Treinamento de modelos para recomendação de tratamentos sem riscos diretos a pacientes.
- **Robótica**: Aprimoramento de controle sem a necessidade de execuções reais.
- **Finanças**: Estratégias de investimento baseadas em dados históricos de mercado.
- **Sistemas de Recomendação**: Aprendizado a partir do comportamento de usuários sem necessidade de testes em tempo real.

---

## Construção de um Dataset para Offline RL

Para ilustrar o funcionamento do Offline RL, utilizamos um **Replay Buffer** contendo transições coletadas de diferentes ambientes do Gymnasium:

- **FrozenLake-v1**
- **Taxi-v3**
- **CliffWalking-v0**
- **CartPole-v1**
- **LunarLander-v3**

Cada entrada no dataset contém:

- Estado atual (`s`)
- Ação tomada (`a`)
- Recompensa recebida (`r`)
- Estado seguinte (`s'`)
- Indicador de finalização (`done`)

A coleta dos dados foi feita utilizando o algoritmo **SARSA**.

```python
import gymnasium as gym
import torch
from util.algorithms import run_sarsa
from util.network import ReplayBuffer

DATASET_SIZE = 200_000
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 1e-3

ENV_NAMES = ["FrozenLake-v1", "Taxi-v3", "CliffWalking-v0"]
ENVS_REPLAY_BUFFER = []

for env_name in ENV_NAMES:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    replay_buffer = ReplayBuffer(DATASET_SIZE, BATCH_SIZE, device)
    env = gym.make(env_name, render_mode="rgb_array")
    
    run_sarsa(env, replay_buffer, DATASET_SIZE, LEARNING_RATE, GAMMA)
    ENVS_REPLAY_BUFFER.append((env_name, env, replay_buffer))
    replay_buffer.save_config(f"config/dataset/sarsa/{env_name}.json")
```

---

## Resultados e Análise

Após treinar o agente com o dataset fixo, avaliamos seu desempenho:

- **FQI** teve bons resultados tanto nos ambientes discretos quanto nos contínuos. No entanto, não conseguiu sucesso no **LunarLander**, possivelmente devido a limitações na rede neural, dados ou hiperparâmetros.
- **CQL** não teve um bom desempenho nos ambientes discretos, indicando que pode precisar de ajustes mais profundos na implementação. Entretanto, teve **bom desempenho no CartPole**.

### Principais Lições

1. **A diversidade dos dados é essencial**: Se o dataset contiver apenas ações subótimas, o agente terá dificuldades em melhorar.
2. **Ajuste fino da rede neural impacta o aprendizado**: Parâmetros como **taxa de aprendizado** e **tamanho do batch** influenciam o desempenho final.
3. **A falta de exploração pode limitar a aprendizagem**: Como o agente não pode testar novas ações, pode ficar preso em políticas subótimas.

---

## Conclusão

O Offline RL representa um avanço significativo para situações onde a coleta de novos dados é custosa ou arriscada. Embora apresente desafios, técnicas como **Fitted Q-Iteration** e **Regularização de Política** podem mitigar problemas comuns.

### Expansão Futura
- **Busca por hiperparâmetros melhores**
- **Melhoria do CQL**
- **Aprendizado por Imitação**
- **Uso de Modelos Generativos**


