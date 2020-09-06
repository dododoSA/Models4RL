from models4rl.replay_buffers.replay_buffer import ReplayBuffer

import torch
import copy



class DQN():
    def __init__(
        self,
        action_space,
        q_network,
        optimizer,
        criterion,
        explorer,
        replay_buffer,
        batch_size=32,
        gamma=0.99,
        target_update_step_interval=0,
        target_update_episode_interval=5
    ):
        assert target_update_step_interval >= 0, 'target_update_step_interval must be positive or 0.'
        assert target_update_episode_interval > 0, 'target_update_episode_interval must be positive.'

        self.q_network = q_network
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.target_network = copy.deepcopy(self.q_network)
        
        self.explorer = explorer

        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        self.gamma = gamma


        self.action_space = action_space
        self.state = None
        self.action = None
        self.target_update_step_interval = target_update_step_interval
        self.target_update_episode_interval = target_update_episode_interval
        self.episode = 1
        self.step = 1

    def act_and_train(self, observation, reward):
        next_state = torch.tensor(observation).float()
        
        self.replay_buffer.append(self.state, self.action, next_state, reward)
        self.replay()

        if self.target_update_step_interval and self.step % self.target_update_step_interval == 0:
            self._update_target_network()

        self.state = next_state
        self.action = self.explorer.explore(self.action_space.sample, lambda: self._choice_greedy_action(next_state))

        self.step += 1
        return self.action

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state_batch, next_state_batch, action_batch, reward_batch = self.replay_buffer.sample(self.batch_size).values()
        

        self.q_network.eval()

        q_values = self.q_network(state_batch).gather(1, action_batch)
        next_q_values = torch.zeros_like(q_values, dtype=float)


        self.target_network.eval()

        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_values = self.gamma * next_q_values + reward_batch


        self.q_network.train()
        
        loss = self.criterion(q_values, target_values.unsqueeze(1).float())
        self.optimizer.zero_grad()
        loss.backward()

        #for p in self.q_network.parameters():
        #    p.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def _update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _choice_greedy_action(self, observation):
        self.q_network.eval()
        with torch.no_grad():
            action = self.q_network(observation).max(0)[1].item()
        return action

    def act_greedy(self, observation):
        next_state = torch.tensor(observation).float()
        return self._choice_greedy_action(next_state)

    def stop_episode_and_train(self, observation, reward):
        next_state = torch.tensor(observation).float()
        self.replay_buffer.append(self.state, self.action, next_state, reward)
        self.replay()
        self.explorer.end_episode()
        
        self.action = None
        self.state = None

        self.episode += 1
        
        if self.episode % self.target_update_episode_interval == 0:
            self._update_target_network()