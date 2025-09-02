import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from config import Config

class QlearningAgent:
    def __init__(self, action_size, mode='train'):
        self.action_size = action_size
        self.mode = mode
        if self.mode == 'train':
            self.learning_rate = Config.LEARNING_RATE
            self.gamma = Config.GAMMA
            self.qtable = {}  # 使用字典存储Q值
            self.epsilon = Config.EPSILON  # 初始探索率
            self.epsilon_min = Config.EPSILON_MIN  # 最小探索率
            self.epsilon_decay = Config.EPSILON_DECAY  # 探索率衰减因子
            
            # 训练记录数据结构
            self.rewards_history = []  # Episode Reward
            self.episode_lengths = []  # Episode Length
            self.epsilon_history = []  # Epsilon (ε)
            self.td_errors = []  # TD-error (每一步)
            self.episode_td_errors = []  # 每回合平均TD-error
            self.q_table_size_history = []  # Q-table Size
            self.training_step_count = 0  # Training Step Count
            self.step_count_history = []  # 每回合步数记录
            
        else:
            self.epsilon = 0.0  # 测试时不探索
            self.qtable = {}
            print("请使用load方法加载以训练好的智能体")

    # 离散化状态空间
    def discretize_state(self, state):
        '''
        description:
            离散化状态空间，因为状态的四个值都是连续的，Q表只能存离散的
        params:
            state: 状态 [cart_pos, cart_vel, pole_angle, pole_ang_vel]
        return:
            d_state: 离散化后的状态 tuple
        '''
        # 设定合理的边界
        bounds = [
            (-2.4, 2.4),    # 小车位置 (比实际范围小一些，避免极端值)
            (-3.0, 3.0),    # 小车速度
            (-0.209, 0.209), # 杆子角度 (约±12度，比±24度更实际)
            (-2.0, 2.0)     # 杆子角速度
        ]
        
        # 每个维度的分箱数量
        n_bins = [10, 10, 10, 10]
        
        discrete_state = []
        for i in range(len(state)):
            # 限制值在边界内
            value = max(bounds[i][0], min(bounds[i][1], state[i]))
            
            # 线性映射到 [0, n_bins[i]-1]
            low, high = bounds[i]
            bin_width = (high - low) / n_bins[i]
            discrete_idx = int((value - low) / bin_width)
            # 防止越界
            discrete_idx = min(discrete_idx, n_bins[i] - 1)
            discrete_idx = max(discrete_idx, 0)
            discrete_state.append(discrete_idx)
        
        return tuple(discrete_state)
        
    def get_action(self, state):
        '''
        description:
            根据state获取动作
                训练模式下是ε-贪婪策略
                测试模式下是纯利用，查表
        params:
            state: 状态
        return:
            action: 动作
        '''
        # 离散化状态
        discrete_state = self.discretize_state(state)
        
        if self.mode == 'test':
            # 测试模式：纯利用，选择Q值最大的动作
            if discrete_state in self.qtable:
                return np.argmax(self.qtable[discrete_state])
            else:
                # 未知状态，随机选择动作
                return np.random.randint(self.action_size)
        
        elif self.mode == 'train':
            # 训练模式：ε-贪婪策略
            if np.random.rand() < self.epsilon:
                # 探索：随机选择动作
                return np.random.randint(self.action_size)
            else:
                # 利用：选择Q值最大的动作
                if discrete_state in self.qtable:
                    return np.argmax(self.qtable[discrete_state])
                else:
                    # 未知状态，随机选择动作
                    return np.random.randint(self.action_size)
    
    def cosine_decay_epsilon(self, episode, total_episodes):
        '''
        description: 
            余弦衰减epsilon
        params:
            episode: 当前回合数
        return:
            None
        '''
        if episode < total_episodes:
            cos_decay = 0.5 * (1 + np.cos(np.pi * episode / total_episodes))
            self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * cos_decay
        else:
            self.epsilon = self.epsilon_min  # 确保在测试时不探索
    def decay_epsilon(self):
        '''
        description:
            衰减epsilon
        params:
            None
        return:
            None
        '''
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def record_episode_data(self, episode_reward, episode_length):
        '''
        description:
            记录每个episode的训练数据
        params:
            episode_reward: 本回合总奖励
            episode_length: 本回合步数
        return:
            None
        '''
        self.rewards_history.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.step_count_history.append(episode_length)
        self.epsilon_history.append(self.epsilon)
        self.q_table_size_history.append(len(self.qtable))
        
        # 计算本回合平均TD-error
        if self.td_errors:
            episode_avg_td = np.mean(self.td_errors[-episode_length:])
            self.episode_td_errors.append(episode_avg_td)

    def update(self, state, action, reward, next_state, done):
        '''
        description:
            更新Q表
        params:
            state: 状态
            action: 动作
            reward: 奖励
            next_state: 下一个状态
            done: 是否结束
        return:
            None
        '''
        # 离散化状态
        discrete_state = self.discretize_state(state)
        next_discrete_state = self.discretize_state(next_state)
        
        # 初始化Q值
        if discrete_state not in self.qtable:
            self.qtable[discrete_state] = np.zeros(self.action_size)
        if next_discrete_state not in self.qtable and not done:
            self.qtable[next_discrete_state] = np.zeros(self.action_size)
        
        # 获取当前Q值
        current_q = self.qtable[discrete_state][action]
        
        # 计算目标Q值
        if done:
            # 终止状态：没有未来奖励
            target_q = reward
        else:
            # 非终止状态：加上折扣的未来奖励
            next_max_q = np.max(self.qtable[next_discrete_state])
            target_q = reward + self.gamma * next_max_q
        
        # 计算TD-error
        td_error = target_q - current_q
        self.td_errors.append(abs(td_error))
        
        # 更新Q值
        self.qtable[discrete_state][action] += self.learning_rate * td_error
        
        # 更新训练步数
        self.training_step_count += 1

    def save(self,path_or_filename):
        '''
        description:
            保存Q表
        params:
            path_or_filename: 保存路径
        return:
            None
        '''
        with open(path_or_filename, 'wb') as f:
            pkl.dump(self.qtable, f)
    def load(self,path_or_filename):
        '''
        description:
            加载Q表
        params:
            path_or_filename: 加载路径
        return:
            None
        '''
        with open(path_or_filename, 'rb') as f:
            self.qtable = pkl.load(f)
        print(f"从 {path_or_filename} 加载Q表完成")

    def plot_training_curves(self, save_path='training_curves.png', window_size=100):
        '''
        description:
            绘制完整的训练曲线，包含所有6个指标，并使用滑动平均平滑
        params:
            save_path: 保存图片的路径
            window_size: 滑动平均窗口大小
        return:
            None
        '''
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Q-Learning Training Curves', fontsize=16, fontweight='bold')
        
        episodes = range(1, len(self.rewards_history) + 1)
        
        # 滑动平均函数
        def moving_average(data, window=window_size):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # 1. Episode Reward (平滑)
        if len(self.rewards_history) >= window_size:
            smoothed_rewards = moving_average(self.rewards_history)
            smooth_episodes = range(window_size, len(self.rewards_history) + 1)
            axes[0, 0].plot(smooth_episodes, smoothed_rewards, color='blue', linewidth=2, label='Smoothed')
            axes[0, 0].plot(episodes, self.rewards_history, color='blue', alpha=0.3, label='Raw')
            axes[0, 0].legend()
        else:
            axes[0, 0].plot(episodes, self.rewards_history, color='blue', alpha=0.7)
        
        axes[0, 0].set_title('Episode Reward (Total Steps per Episode)', fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Episode Length (平滑)
        if len(self.episode_lengths) >= window_size:
            smoothed_lengths = moving_average(self.episode_lengths)
            smooth_episodes = range(window_size, len(self.episode_lengths) + 1)
            axes[0, 1].plot(smooth_episodes, smoothed_lengths, color='green', linewidth=2, label='Smoothed')
            axes[0, 1].plot(episodes, self.episode_lengths, color='green', alpha=0.3, label='Raw')
            axes[0, 1].legend()
        else:
            axes[0, 1].plot(episodes, self.episode_lengths, color='green', alpha=0.7)
        
        axes[0, 1].set_title('Episode Length (Steps per Episode)', fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Epsilon (ε) - 不需要平滑
        axes[1, 0].plot(episodes, self.epsilon_history, color='red', linewidth=2)
        axes[1, 0].set_title('Epsilon (Exploration Rate)', fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. TD-error (每回合平均) (平滑)
        if self.episode_td_errors:
            td_errors = [abs(x) for x in self.episode_td_errors]  # 取绝对值
            if len(td_errors) >= window_size:
                smoothed_td = moving_average(td_errors)
                smooth_episodes = range(window_size, len(td_errors) + 1)
                axes[1, 1].plot(smooth_episodes, smoothed_td, color='orange', linewidth=2, label='Smoothed')
                axes[1, 1].plot(episodes, td_errors, color='orange', alpha=0.3, label='Raw')
                axes[1, 1].legend()
            else:
                axes[1, 1].plot(episodes, td_errors, color='orange', alpha=0.7)
            
            axes[1, 1].set_title('TD-error (Average per Episode)', fontweight='bold')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average |TD-error|')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')  # 对数尺度，便于观察小误差
        
        # 5. Q-table Size
        axes[2, 0].plot(episodes, self.q_table_size_history, color='purple', linewidth=2)
        axes[2, 0].set_title('Q-table Size (Learned States)', fontweight='bold')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Number of States')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Training Step Count (累积)
        cumulative_steps = np.cumsum(self.step_count_history)
        axes[2, 1].plot(episodes, cumulative_steps, color='brown', linewidth=2)
        axes[2, 1].set_title('Training Step Count (Cumulative)', fontweight='bold')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Total Steps')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"训练曲线已保存到: {save_path}")

    def get_training_summary(self):
        '''
        description:
            获取训练摘要信息
        return:
            dict: 包含所有训练指标的字典
        '''
        return {
            'total_episodes': len(self.rewards_history),
            'total_steps': self.training_step_count,
            'final_epsilon': self.epsilon_history[-1] if self.epsilon_history else 0,
            'final_q_table_size': self.q_table_size_history[-1] if self.q_table_size_history else 0,
            'average_reward_last_100': np.mean(self.rewards_history[-100:]) if len(self.rewards_history) >= 100 else np.mean(self.rewards_history),
            'max_reward': max(self.rewards_history) if self.rewards_history else 0,
            'average_td_error': np.mean(self.episode_td_errors) if self.episode_td_errors else 0
        }