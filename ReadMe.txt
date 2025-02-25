# Offline Reinforcement Learning

Este projeto explora o conceito de **Offline Reinforcement Learning (Offline RL)**, onde um agente aprende a partir de um dataset fixo de experiências pré-coletadas, sem interagir diretamente com o ambiente durante o treinamento. O objetivo é estudar e implementar algoritmos de Offline RL, comparando seu desempenho em diferentes ambientes.

## Tema

**Tema 12: Offline Reinforcement Learning**

**Pergunta:**  
Como um agente pode aprender a partir de um dataset fixo de experiências pré-coletadas (e não de forma on-line, coletando dados à medida que aprende)?

## Objetivos

- Estudar e implementar algoritmos de Offline RL.
- Coletar datasets de experiências em ambientes como `FrozenLake-v1`, `Taxi-v3` e `CliffWalking-v0`.
- Treinar e comparar modelos de Offline RL (CQL-DQN e FQI-DQN) usando os datasets coletados.
- Analisar os resultados e discutir as implicações práticas do Offline RL.

## Requisitos

- Python 3.8 ou superior.
- Bibliotecas necessárias: `gymnasium`, `torch`, `numpy`, `matplotlib`, `argparse`.
- Um ambiente de execução com poder computacional similar ao Google Colab (recomendado).
