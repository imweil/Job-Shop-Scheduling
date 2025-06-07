import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from src.agent.net import Encoder, Decoder, ValueNet


class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder()
        self.time_agent = Decoder(out_size=10)
        self.pattern_agent = Decoder(out_size=5, embed_size=96)

        self.window_critic = ValueNet()
        self.pattern_critic = ValueNet(state_dim=96)

        self.gamma = 0.99
        self.eps = 0.2
        self.epochs = 10
        self.lmbda = 0.95

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.001)
        self.window_optimizer = torch.optim.Adam(self.time_agent.parameters(), lr=0.001)
        self.pattern_optimizer = torch.optim.Adam(self.pattern_agent.parameters(), lr=0.001)
        self.window_critic_optimizer = torch.optim.Adam(self.window_critic.parameters(), lr=0.001)
        self.pattern_critic_optimizer = torch.optim.Adam(self.pattern_critic.parameters(), lr=0.001)

    def take_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0)
            encoder_e = torch.cat([state_tensor, torch.full((1, 3), 0)], dim=1)
            embedding = self.encoder(encoder_e)
            embedding = embedding.view(embedding.size(0), -1)
            window_probs = self.time_agent(embedding)

            window_dist = torch.distributions.Categorical(window_probs)
            window = window_dist.sample().item()

            pattern_embedding = torch.cat([embedding, torch.full((1, 8), window)], dim=1)
            scheduling_pattern_probs = self.pattern_agent(pattern_embedding)

            window_probs = window_probs.view(-1)
            scheduling_pattern_probs = scheduling_pattern_probs.view(-1)

            scheduling_pattern = torch.distributions.Categorical(scheduling_pattern_probs)
            scheduling_pattern = scheduling_pattern.sample().item()

            window = 0
            window_probs=0
        return window, scheduling_pattern, window_probs, scheduling_pattern_probs, pattern_embedding

    def update_windows(self, transition_dict):
        self.encoder.to(self.device)
        self.time_agent.to(self.device)
        self.window_critic.to(self.device)

        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        states3 = torch.cat([states, torch.full((states.shape[0], 3), 0, device=self.device)], dim=1)
        next_states3 = torch.cat([next_states, torch.full((states.shape[0], 3), 0, device=self.device)], dim=1)

        td_target = rewards + self.gamma * self.window_critic(next_states3) * (1 - dones)
        td_delta = td_target - self.window_critic(states3)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        old_log_probs = torch.log(
            self.time_agent(self.encoder(states3)[:, 0, :].unsqueeze(1)).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.time_agent(self.encoder(states3)[:, 0, :].unsqueeze(1)).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.window_critic(states3), td_target.detach()))
            self.window_optimizer.zero_grad()
            self.window_critic_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.window_optimizer.step()
            self.window_critic_optimizer.step()
            self.encoder_optimizer.step()

        self.encoder.to('cpu')
        self.time_agent.to('cpu')
        self.window_critic.to('cpu')

    def update_pattern(self, transition_dict):
        self.pattern_agent.to(self.device)
        self.pattern_critic.to(self.device)

        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.pattern_critic(next_states) * (1 - dones)
        td_delta = td_target - self.pattern_critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.pattern_agent(states).gather(1, actions)).detach()

        for epoch in range(self.epochs):
            log_probs = torch.log(self.pattern_agent(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.pattern_critic(states), td_target.detach()))

            self.pattern_critic_optimizer.zero_grad()
            self.pattern_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            self.pattern_critic_optimizer.step()
            self.pattern_optimizer.step()

        self.pattern_agent.to('cpu')
        self.pattern_critic.to('cpu')

    def save(self, i):
        filepath = "../model/agent_model" + i + ".pth"
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'time_agent_state_dict': self.time_agent.state_dict(),
            'pattern_agent_state_dict': self.pattern_agent.state_dict(),
        }, filepath)

    def load(self, i):
        filepath = "../model/agent_model" + i + ".pth"
        checkpoint = torch.load(filepath)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.time_agent.load_state_dict(checkpoint['time_agent_state_dict'])
        self.pattern_agent.load_state_dict(checkpoint['pattern_agent_state_dict'])


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
