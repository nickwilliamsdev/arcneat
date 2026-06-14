import torch
from src.env.arc_env import ARCEnv
from src.models.deltanet import DeltaNetARCModel
from src.agents.muzero import MuZeroAgent
from src.agents.tensorneat import TensorNeatRunner

def main():
    print("Initializing ARC Environment (ArcAGI2 framework)...")
    env = ARCEnv(max_steps=10)
    
    print("Environment created. Observation space:", env.observation_space.shape)
    
    obs, info = env.reset()
    
    print(f"Task ID: {info['task_id']}")
    print(f"Initial Observation Shape: {obs.shape}")

    # Set up DeltaNet
    vocab_size = 10  # 10 distinctive colors in ARC
    d_model = 64
    action_space = env.action_space.n

    print(f"Initializing DeltaNet and MuZero Agent for sequence reasoning...")
    muzero = MuZeroAgent(vocab_size=vocab_size, d_model=d_model, action_space=action_space)

    print("Running initial MuZero inference...")
    # Flat representation of grid
    flat_obs = torch.tensor(obs.flatten(), dtype=torch.long).unsqueeze(0) # (1, L)
    
    hidden, policy, value = muzero.initial_inference(flat_obs)
    print(f"Policy logits shape: {policy.shape}, Value estimate: {value.item():.4f}")

    print("Initializing PyTorch-TensorNEAT Runner...")
    neat_runner = TensorNeatRunner(
        DeltaNetARCModel, 
        dict(vocab_size=vocab_size, d_model=d_model, action_space=action_space),
        population_size=5
    )
    
    print("Compiling and evaluating TensorNEAT individual 0...")
    neat_policy, neat_val = neat_runner.evaluate_individual(0, flat_obs)
    print(f"NEAT evaluated successfully. Value: {neat_val.item():.4f}")
    
    print("\nSetup complete. torcharc is structured to run MuZero integrated with PyTorch-TensorNEAT and DeltaNets on ARC tasks.")

if __name__ == "__main__":
    main()
