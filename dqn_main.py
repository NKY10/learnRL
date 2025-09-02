#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from DQNagent import DQNAgent
from config import DQNConfig

def train_agent():
    """训练智能体"""
    env = gym.make('CartPole-v1')
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        DQNConfig=DQNConfig,
        mode="train"
    )
    
    # 训练记录数据结构
    rewards_history = []  # Episode Reward
    episode_lengths = []  # Episode Length
    epsilon_history = []  # Epsilon (ε)
    episode_losses = []   # 每回合平均损失
    training_step_count = 0  # Training Step Count
    step_count_history = []  # 每回合步数记录
    
    for episode in range(DQNConfig.EPISODES):
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        losses = []
        
        while step_count < DQNConfig.MAX_STEPS:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
                        
            loss = agent.update_local(state, action, reward, next_state, terminated)
            losses.append(loss)
            
            state = next_state
            total_reward += reward
            step_count += 1
            training_step_count += 1
            
            # 定期更新目标网络
            if training_step_count % agent.step_update_target == 0:
                agent.update_target()
            
            if terminated or truncated:
                break
        agent.cosine_decay_epsilon(episode, DQNConfig.EPISODES)
        rewards_history.append(total_reward)
        episode_lengths.append(step_count)
        step_count_history.append(step_count)
        epsilon_history.append(agent.epsilon)
        
        # 记录平均损失
        avg_loss = np.mean(losses) if losses else 0
        episode_losses.append(avg_loss)
        
        # 使用余弦退火衰减epsilon
        agent.cosine_decay_epsilon(episode, DQNConfig.EPISODES)
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{DQNConfig.EPISODES}, Reward: {total_reward}, Steps: {step_count}, Epsilon: {agent.epsilon:.4f}, Avg Loss: {avg_loss:.6f}")
            
            # 打印一些统计信息
            if len(agent.memory) > 0:
                recent_losses = [loss for loss in losses[-10:] if loss > 0]
                if recent_losses:
                    #print(f"  Recent losses (last 10): {recent_losses[-5:]}")
                    print(f"  Memory buffer size: {len(agent.memory)}")
    
    agent.save(DQNConfig.MODEL_PATH)
    env.close()
    
    # 打印训练摘要
    print({
        'total_episodes': len(rewards_history),
        'total_steps': training_step_count,
        'final_epsilon': epsilon_history[-1] if epsilon_history else 0,
        'average_reward_last_100': np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history),
        'max_reward': max(rewards_history) if rewards_history else 0
    })
    
    # 绘制训练曲线
    plot_training_curves(rewards_history, episode_lengths, epsilon_history, episode_losses, step_count_history)


def plot_training_curves(rewards_history, episode_lengths, epsilon_history, episode_losses, step_count_history, save_path='dqn_training_curves.png', window_size=100):
    '''绘制训练曲线'''
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('DQN Training Curves', fontsize=16, fontweight='bold')
    
    episodes = range(1, len(rewards_history) + 1)
    
    # 滑动平均函数
    def moving_average(data, window=window_size):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # 1. Episode Reward (平滑)
    if len(rewards_history) >= window_size:
        smoothed_rewards = moving_average(rewards_history)
        smooth_episodes = range(window_size, len(rewards_history) + 1)
        axes[0, 0].plot(smooth_episodes, smoothed_rewards, color='blue', linewidth=2, label='Smoothed')
        axes[0, 0].plot(episodes, rewards_history, color='blue', alpha=0.3, label='Raw')
        axes[0, 0].legend()
    else:
        axes[0, 0].plot(episodes, rewards_history, color='blue', alpha=0.7)
    
    axes[0, 0].set_title('Episode Reward (Total Steps per Episode)', fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Episode Length (平滑)
    if len(episode_lengths) >= window_size:
        smoothed_lengths = moving_average(episode_lengths)
        smooth_episodes = range(window_size, len(episode_lengths) + 1)
        axes[0, 1].plot(smooth_episodes, smoothed_lengths, color='green', linewidth=2, label='Smoothed')
        axes[0, 1].plot(episodes, episode_lengths, color='green', alpha=0.3, label='Raw')
        axes[0, 1].legend()
    else:
        axes[0, 1].plot(episodes, episode_lengths, color='green', alpha=0.7)
    
    axes[0, 1].set_title('Episode Length (Steps per Episode)', fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Epsilon (ε) - 不需要平滑
    axes[1, 0].plot(episodes, epsilon_history, color='red', linewidth=2)
    axes[1, 0].set_title('Epsilon (Exploration Rate)', fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Loss (每回合平均) (平滑)
    if episode_losses and any(episode_losses):
        if len(episode_losses) >= window_size:
            smoothed_losses = moving_average(episode_losses)
            smooth_episodes = range(window_size, len(episode_losses) + 1)
            axes[1, 1].plot(smooth_episodes, smoothed_losses, color='orange', linewidth=2, label='Smoothed')
            axes[1, 1].plot(episodes, episode_losses, color='orange', alpha=0.3, label='Raw')
            axes[1, 1].legend()
        else:
            axes[1, 1].plot(episodes, episode_losses, color='orange', alpha=0.7)
        
        axes[1, 1].set_title('Loss (Average per Episode)', fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Loss')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')  # 对数尺度，便于观察小误差
    
    # 5. Training Step Count (累积)
    if step_count_history:
        cumulative_steps = np.cumsum(step_count_history)
        axes[2, 0].plot(episodes, cumulative_steps, color='brown', linewidth=2)
        axes[2, 0].set_title('Training Step Count (Cumulative)', fontweight='bold')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Total Steps')
        axes[2, 0].grid(True, alpha=0.3)
    
    # 移除空的子图
    axes[2, 1].remove()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"训练曲线已保存到: {save_path}")


def test_agent(save_gif=True, gif_path="dqn_test_episode.gif"):
    """测试智能体，并可选地保存一轮的运行过程为GIF"""
    # 使用 rgb_array 模式获取图像帧
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        DQNConfig=DQNConfig,
        mode="test"
    )
    
    total_rewards = []
    
    # 用于保存图像帧
    frames = []

    for episode in range(10):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 渲染当前帧（仅在主测试轮次中记录第一轮）
            if save_gif and len(total_rewards) == 0:  # 只记录第一轮为GIF
                frame = env.render()
                frames.append(Image.fromarray(frame))

            action = agent.get_action(state)
            result = env.step(action)
            state, reward, terminated, truncated, _ = result
            total_reward += reward

            done = terminated or truncated

        total_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Reward = {total_reward}")
        
        if done and len(total_rewards) == 1 and save_gif:
            print("Saving GIF...")
            os.makedirs(os.path.dirname(gif_path) if os.path.dirname(gif_path) else '.', exist_ok=True)
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=50,  # 每帧50毫秒，约20fps
                loop=0  # 0表示无限循环
            )
            print(f"GIF saved to {gif_path}")

    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_agent()
    else:
        train_agent()