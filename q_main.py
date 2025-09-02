#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
from Qagent import QlearningAgent
from config import Config

def train_agent():
    """训练智能体"""
    env = gym.make('CartPole-v1')
    agent = QlearningAgent(
        action_size=env.action_space.n
    )
    
    for episode in range(Config.EPISODES):
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        
        while step_count < Config.MAX_STEPS:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.update(state, action, reward, next_state, terminated)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        agent.record_episode_data(total_reward, step_count)
        agent.cosine_decay_epsilon(episode, Config.EPISODES)
        #agent.decay_epsilon()
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{Config.EPISODES}, Reward: {total_reward}")
    
    agent.save(Config.MODEL_PATH)
    env.close()
    
    print(agent.get_training_summary())
    agent.plot_training_curves()


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from config import Config  # 假设你的配置文件路径

def test_agent(save_gif=True, gif_path="test_episode.gif"):
    """测试智能体，并可选地保存一轮的运行过程为GIF"""
    # 使用 rgb_array 模式获取图像帧
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    agent = QlearningAgent(action_size=2, mode='test')
    agent.load(Config.MODEL_PATH)
    
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
