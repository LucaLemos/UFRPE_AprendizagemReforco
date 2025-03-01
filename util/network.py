import json
import random
import torch
import numpy as np
from collections import deque, namedtuple
from torch import optim, nn

def to_one_hot(size, states): 
    states = torch.tensor(states, dtype=torch.long)  # Garante que é tensor de inteiros
    one_hot = torch.zeros(len(states), size)  # Cria matriz de zeros
    one_hot[torch.arange(len(states)), states] = 1  # Marca os índices com 1
    return one_hot

def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, memory=None, experience_fields=None):
        """Initialize a ReplayBuffer object."""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Definição da experiência
        self.experience = namedtuple("Experience", field_names=experience_fields or ["state", "action", "reward", "done", "next_state"])

        # Inicializa a memória (carregar caso já tenha sido salvo antes)
        self.memory = deque(
            (self.experience(
                state=exp["state"],
                action=np.int64(exp["action"]),  # Convertendo para np.int64
                reward=exp["reward"],
                next_state=exp["next_state"],
                done=exp["done"]
            ) for exp in memory),
            maxlen=buffer_size
        ) if memory else deque(maxlen=buffer_size)

    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample_continuous(self):
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.memory[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

    def save_config(self, filename):
        """Salva a configuração da ReplayBuffer em um arquivo JSON."""
        print()
        #data = [{k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v for k, v in exp._asdict().items()} for exp in self.memory]
        data = [
            {
                k: int(v) if isinstance(v, np.integer) else 
                float(v) if isinstance(v, np.floating) else 
                v.tolist() if isinstance(v, np.ndarray) else 
                v
                for k, v in exp._asdict().items()
            } 
            for exp in self.memory
        ]
        
        config = {
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "device": self.device.type,
            "experience_fields": self.experience._fields,
            "memory": data  # Convertendo namedtuple para dicionário
        }
        with open(filename, "w") as f:
            json.dump(config, f, indent=4)
    
    @classmethod
    def load_config(cls, filename):
        """Carrega a configuração da ReplayBuffer de um arquivo JSON."""
        with open(filename, "r") as f:
            config = json.load(f)
        return cls(
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            device=config["device"],
            memory=config["memory"],
            experience_fields=config["experience_fields"]
        )
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, list_hidden_dims, final_activ_fn=None):
        super().__init__()
        self.state_size = input_dim
        self.action_size = output_dim
        layers = []
        last_dim = input_dim
        for dim in list_hidden_dims:
            layers.append(nn.Linear(last_dim, dim, bias=True))
            layers.append(nn.ReLU())
            last_dim = dim
        layers.append(nn.Linear(last_dim, output_dim, bias=True))
        if final_activ_fn is not None:
            layers.append( final_activ_fn )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

class Agent():
    def __init__(self, state_size, action_size, tau, gamma, lr, hidden_size, device="cpu", isDiscrete=True):
        self.isDiscrete = isDiscrete
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.tau = tau
        self.gamma = gamma
        self.lr = lr
        self.network = MLP(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_net = MLP(self.state_size, self.action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr)


    def get_action(self, state):
        state_tensor = torch.tensor([state], dtype=torch.float)  # Criar um tensor de batch_size=1
        state = to_one_hot(self.state_size, state_tensor)[0].float().unsqueeze(0).to(self.device)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        return action

    def save(self, args, save_name):
        import os
        save_dir = './trained_models/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.network.state_dict(), save_dir + args.env + save_name + ".pth")

    def soft_update(self):
        for target_param, local_param in zip(self.target_net.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def bellman_error(self, experiences):
        
        
        states, actions, rewards, dones, next_states = experiences
        #print(f"Actions Antes: {actions}")
        #print(f"States Antes: {states}")
        #print(f"Dones Antes: {dones}")
        #print(f"next_States Antes: {next_states}")
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones, dtype=torch.bool)
        #print(f"Actions Depois: {actions}")
        #print(f"Dones Depois: {dones}")
        if self.isDiscrete:
            states = to_one_hot(self.state_size, states)
            next_states = to_one_hot(self.state_size, next_states)
        else:
            states = torch.tensor(states, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
        #print(f"States Depois: {states}")
        with torch.no_grad():
            # Calcula Q_targets apenas com a equação de Bellman
            #print(f"next_states: {next_states}")
            #print(f"target_net: {self.target_net}")
            Q_targets_next = self.target_net(next_states).max(dim=1)[0]
            #print(f"Q_targets_Next: {Q_targets_next}")
            Q_targets_next[dones] = 0.0
            Q_targets_next = Q_targets_next.detach()
            if self.isDiscrete:
                Q_targets_next = Q_targets_next.unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next)
            
        Q_a_s = self.network(states)
        #print(f"Q_a_s: {Q_a_s}")

        Q_expected = Q_a_s.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        # Remove o termo CQL, deixando apenas a loss de Bellman
        loss_fn = nn.SmoothL1Loss()  # Huber Loss (melhor que MSE para estabilidade)
        bellman_error = loss_fn(Q_expected, Q_targets)
        return bellman_error, Q_a_s, actions

class CQLAgent(Agent):
    def __init__(self, state_size, action_size, tau, gamma, lr, alpha, hidden_size=[128, 128], device="cpu", isDiscrete=True):
        super().__init__(state_size, action_size, tau, gamma, lr, hidden_size, device, isDiscrete)
        self.alpha = alpha
        self.isDiscrete = isDiscrete
        
    def cql_loss(self, q_values, current_action):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action.unsqueeze(-1)).squeeze(-1)
        return (logsumexp - q_a).mean()
    
    def learn(self, experiences):
        self.network.train()
        bellman_error, Q_a_s, actions = self.bellman_error(experiences)
        
        cql1_loss = self.cql_loss(Q_a_s, actions)
        q1_loss = cql1_loss + self.alpha * bellman_error

        self.optimizer.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update()
        return q1_loss.detach().item(), cql1_loss.detach().item(), bellman_error.detach().item()

class FQIAgent(Agent):
    def __init__(self, state_size, action_size, tau, gamma, lr, model, hidden_size=256, device="cpu"):
        super().__init__(state_size, action_size, tau, gamma, lr, model, hidden_size, device)
    
    def learn(self, experiences):
        self.network.train()
        bellman_error, _, _ = self.bellman_error(experiences)
        
        self.optimizer.zero_grad()
        bellman_error.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        # Atualiza a rede-alvo de forma suave
        self.soft_update()

        return bellman_error.detach().item()
    
#   -------------------------------------------     EXTRA      -------------------------------------------------

# Faz uma escolha epsilon-greedy
def epsilon_greedy_qnet(qnet, env, state, epsilon):
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        state_v = torch.tensor(state, dtype=torch.float32)
        state_v = state_v.unsqueeze(0)  # Adiciona dimensão de batch como eixo 0 (e.g. transforma uma lista [a,b,c] em [[a,b,c]])
        q_vals_v = qnet(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.item())
    return action

# loss function, para treinamento da rede no DQN
def calc_loss(batch, net, tgt_net, gamma):
    states, actions, rewards, dones, next_states = batch
    #print(f"next_states Antes: {next_states}")
    states_v = torch.tensor(states, dtype=torch.float32)
    next_states_v = torch.tensor(next_states, dtype=torch.float32)
    actions_v = torch.tensor(actions, dtype=torch.int64)
    rewards_v = torch.tensor(rewards)
    done_mask = torch.tensor(dones, dtype=torch.bool)

    #print(f"next_states Antes: {next_states}")
    #print(f"net: {net}")
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(dim=1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    target_state_action_values = rewards_v + gamma * next_state_values
    return nn.MSELoss()(state_action_values, target_state_action_values)