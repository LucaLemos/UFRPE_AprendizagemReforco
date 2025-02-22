import json
import random
import torch
import wandb
import numpy as np
from collections import deque, namedtuple
from torch import optim, nn

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, memory=None, experience_fields=None):
        """Initialize a ReplayBuffer object."""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Definição da experiência
        self.experience = namedtuple("Experience", field_names=experience_fields or ["state", "action", "reward", "next_state", "done"])

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

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def save_config(self, filename):
        """Salva a configuração da ReplayBuffer em um arquivo JSON."""
        data = [{k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v for k, v in exp._asdict().items()} for exp in self.memory]
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

class DDQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size):
        super(DDQN, self).__init__()
        self.input_size = state_size
        self.action_size = action_size
        print(f"Input Size: {self.input_size} (type: {type(self.input_size)})")
        print(f"Layer Size: {layer_size} (type: {type(layer_size)})")

        self.head_1 = nn.Linear(self.input_size, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)

    def forward(self, input):
        """

        """
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)

        return out

class CQLAgent():
    def __init__(self, state_size, action_size, tau, gamma, lr, hidden_size=256, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.tau = tau
        self.gamma = gamma

        self.network = DDQN(self.state_size, self.action_size, hidden_size).to(self.device)

        self.target_net = DDQN(self.state_size, self.action_size, hidden_size).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr)


    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state_tensor = torch.tensor([state], dtype=torch.long)  # Criar um tensor de batch_size=1
            state = to_one_hot(state_tensor, self.state_size)[0].float().unsqueeze(0).to(self.device)
            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            self.network.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            action = random.choices(np.arange(self.action_size), k=1)
            #action = np.array([random.randint(0, self.action_size - 1)])
        return action

    def cql_loss(self, q_values, current_action):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)

        return (logsumexp - q_a).mean()

    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences
        states = to_one_hot(states, self.state_size)
        next_states = to_one_hot(next_states, self.state_size)
        #print(next_states)
        with torch.no_grad():
            Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_a_s = self.network(states)
        Q_expected = Q_a_s.gather(1, actions)

        cql1_loss = self.cql_loss(Q_a_s, actions)

        loss_fn = nn.MSELoss()  # Criando a função de perda
        bellman_error = loss_fn(Q_expected, Q_targets)

        q1_loss = cql1_loss + 0.5 * bellman_error

        self.optimizer.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_net)
        return q1_loss.detach().item(), cql1_loss.detach().item(), bellman_error.detach().item()


    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            
def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        #wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        #wandb.save(save_dir + args.run_name + save_name + ".pth")

def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()

def to_one_hot(states, state_size): 
    one_hot_states = torch.zeros((states.shape[0], state_size), device=states.device)
    one_hot_states.scatter_(1, states.long().unsqueeze(1), 1)  
    return one_hot_states