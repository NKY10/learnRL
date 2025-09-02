使用 Q-Learning 和深度 Q 网络（DQN）两种方法解决cartpole

```
├── Qagent.py              # Q-Learning 智能体实现
├── DQNagent.py            # 深度 Q 网络智能体实现
├── q_main.py              # Q-Learning 训练和测试主脚本
├── dqn_main.py            # DQN 训练和测试主脚本
├── config.py              # 两种算法的配置参数
├── cartpole_q_table.pkl   # 训练好的 Q 表（Q-Learning）
├── cartpole_dqn_model.pth # 训练好的 DQN 模型
├── training_curves.png    # Q-Learning 训练性能图
├── dqn_training_curves.png # DQN 训练性能图
├── test_episode.gif       # Q-Learning 智能体演示
└── dqn_test_episode.gif   # DQN 智能体演示
```

Q-learning
**训练：**
```bash
python q_main.py
```

**测试：**
```bash
python q_main.py test
```
DQN
**训练：**
```bash
python dqn_main.py
```

**测试：**
```bash
python dqn_main.py test
```

关键参数可以在 `config.py` 中修改：

**Q-Learning 参数 （四个状态均离散化为10个状态）**

- `EPISODES`: 训练回合数（默认：50,000）
- `LEARNING_RATE`: Q 表更新学习率（默认：0.1）
- `GAMMA`: 折扣因子（默认：0.9999）
- `EPSILON`: 初始探索率（默认：1.0）
- `EPSILON_MIN`: 最小探索率（默认：0.01）
- `EPSILON_DECAY`: 探索衰减率（默认：0.995）
- `MAX_STEPS`: 每回合最大步数（默认：500）

**DQN参数**

- `EPISODES`: 训练回合数（默认：800）
- `LEARNING_RATE`: 神经网络学习率（默认：5e-4）
- `BATCH_SIZE`: 经验回放批量大小（默认：64）
- `MEMORY_SIZE`: 经验回放缓冲区大小（默认：10,000）
- `STEP_UPDATE_TARGET`: 目标网络更新频率（默认：100）


---



实验中**探索率调度策略**对训练过程影响显著。

相比传统的指数衰减，余弦退火方法表现更优。这是因为余弦衰减在训练初期保持较高的探索率更久，使 agent 能充分探索环境，访问更多状态-动作对，从而更全面地更新 Q 值。这有助于避免过早收敛到次优策略，加快整体收敛速度。

🔑 Exploration（探索） vs Exploitation（利用）是强化学习中最基本、最深刻的权衡问题。 

Q-learning训练过程示例
LEARNING_RATE = 0.1，EPISODES = 5000

指数衰减：

![cd287d39-a02f-4e47-8722-527884308e34](/Volumes/sn580/projects/carpole/assets/cd287d39-a02f-4e47-8722-527884308e34.png)

余弦衰减：

![981efb58-cece-4fb6-9277-9690e38a3c2a](/Volumes/sn580/projects/carpole/assets/981efb58-cece-4fb6-9277-9690e38a3c2a.png)
