import torch
import numpy as np
import random
from collections import deque
import pickle as pkl
from config import DQNConfig



class QNetwork(torch.nn.Module):
    '''
        价值函数的神经网络，输入状态，输出每个动作的价值
    '''
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size, DQNConfig, mode="train"):
        self.action_size = action_size
        self.state_size = state_size
        self.mode = mode
        if self.mode == 'train':
            self.qnetwork_main = QNetwork(state_size, action_size) # 主网络
            self.qnetwork_target = QNetwork(state_size, action_size) # 目标网络
            # 初始时刻要两个网络同步，否则随机初始化的，一开始预测不准，导致 target_q_values 不稳定。
            self.qnetwork_target.load_state_dict(self.qnetwork_main.state_dict())
            # 模型训练参数
            self.optimizer = torch.optim.Adam(self.qnetwork_main.parameters(), lr=DQNConfig.LEARNING_RATE)
            self.memory = ExperienceReplay(DQNConfig.MEMORY_SIZE)
            self.step_update_target = DQNConfig.STEP_UPDATE_TARGET
            self.gamma = DQNConfig.GAMMA
            self.epsilon = DQNConfig.EPSILON
            self.epsilon_min = DQNConfig.EPSILON_MIN
            self.batch_size = DQNConfig.BATCH_SIZE
        elif self.mode == 'test':
            self.qnetwork_target = self.load(DQNConfig.MODEL_PATH)
        else:
            raise ValueError("mode must be 'train' or 'test'")
    
    def normalize_state(self, state):
        """归一化状态"""
        return state
        #return (state - self.state_mean) / (self.state_std + 1e-8)  # 添加小值防止除零

    def get_action(self, state):
        '''
            选择动作
        '''
        # 归一化状态
        normalized_state = self.normalize_state(state)
        
        if self.mode == 'train':
            if np.random.rand() <= self.epsilon:
                # 探索，随机选择动作
                return np.random.choice(self.action_size)
            else:
                # 利用，选择Q值最大的动作
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0)
                    q_values = self.qnetwork_main(state_tensor)
                    return torch.argmax(q_values).item()
        elif self.mode == 'test':
            with torch.no_grad():
                state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0)
                q_values = self.qnetwork_target(state_tensor)
                return torch.argmax(q_values).item()


    def update_local(self, state, action, reward, next_state, done):
        '''
            更新网络
        '''
        if self.mode == 'train':
            # 归一化状态
            normalized_state = self.normalize_state(state)
            normalized_next_state = self.normalize_state(next_state)
            
            # 存储经验
            self.memory.push(normalized_state, action, reward, normalized_next_state, done)
            
            # 如果经验回放缓冲区中的经验数量小于batch_size，则不进行训练
            if len(self.memory) < self.batch_size:
                return 0
            
            # 从经验回放缓冲区中采样
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            # 转换为tensor
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)
            
            # 计算当前Q值
            current_q_values = self.qnetwork_main(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # 计算目标Q值
            with torch.no_grad():
                next_q_values = self.qnetwork_target(next_states).max(1)[0]
                # 当episode结束时，目标Q值就是当前奖励
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # 计算损失 - mse
            loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
            
            # 更新网络
            self.optimizer.zero_grad()
            loss.backward()
            
            # 添加梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.qnetwork_main.parameters(), max_norm=1.0)
            
            self.optimizer.step()
                        
            return loss.item()
        return 0

    def update_target(self):
        '''
            更新目标网络
        '''
        if self.mode == 'train':
            self.qnetwork_target.load_state_dict(self.qnetwork_main.state_dict())
        else:
            raise ValueError("训练模式才能更新")
    def save(self, path):
        '''
            保存模型
        '''
        torch.save(self.qnetwork_target.state_dict(), path)
    
    def load(self, path):
        '''
            加载模型
        '''
        model = QNetwork(self.state_size, self.action_size)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def cosine_decay_epsilon(self, episode, total_episodes):
        '''
            余弦退火衰减epsilon
        '''
        if self.mode == 'train':
            if episode < total_episodes:
                cos_decay = 0.5 * (1 + np.cos(np.pi * episode / total_episodes))
                self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * cos_decay
            else:
                self.epsilon = self.epsilon_min


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        '''
            存储经验
        '''
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        '''
            随机采样经验
        '''
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        '''
            返回经验回放缓冲区的大小
        '''
        return len(self.buffer)
