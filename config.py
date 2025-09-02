class QConfig:
    EPISODES = 50000
    LEARNING_RATE = 0.1
    GAMMA = 0.9999
    EPSILON = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    MAX_STEPS = 500
    MODEL_PATH = 'cartpole_q_table.pkl'


class DQNConfig:
    EPISODES = 800            
    LEARNING_RATE = 5e-4      
    
    GAMMA = 0.99              
    EPSILON = 1.0             
    EPSILON_MIN = 0.01        
    
    MAX_STEPS = 500           
    STEP_UPDATE_TARGET = 100  
    BATCH_SIZE = 64           
    MEMORY_SIZE = 10000       
    MODEL_PATH = 'cartpole_dqn_model.pth' 
