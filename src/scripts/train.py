import os
import torch
import argparse
import copy
from tqdm import tqdm
from src.env.arc_env import ARCEnv
from src.models.deltanet import DeltaNetARCModel
from src.agents.tensorneat import TensorNeatRunner

def evaluate_individual(model, env, num_episodes=3, device="cpu"):
    """
    Evaluates a single model in the environment across multiple episodes.
    """
    total_reward = 0.0
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        state = None # Recurrent state for the DeltaNet
        
        while not done:
            # Flatten observation to pass into the model
            flat_obs = torch.tensor(obs.flatten(), dtype=torch.long, device=device).unsqueeze(0)
            
            with torch.no_grad():
                # Forward pass: extract action probabilities for the grid
                policy_logits, value, state = model(flat_obs, state)
                # Greedy action selection (argmax)
                action = policy_logits.argmax(dim=-1).item()
            
            # Step the arc environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if truncated:
                break
                
    return total_reward / num_episodes

def main():
    parser = argparse.ArgumentParser(description="Kickoff Torcharc Evolutionary Run")
    parser.add_argument("--generations", type=int, default=5000, help="Number of evolutionary iterations")
    parser.add_argument("--pop-size", type=int, default=32, help="Population size for TensorNEAT")
    parser.add_argument("--episodes", type=int, default=5, help="Number of ARC tasks evaluated per individual")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument("--ckpt-freq", type=int, default=50, help="Save a checkpoint every N generations")
    args = parser.parse_args()

    # Ensure checkpoint directory exists
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # Detect GPU hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing environment...")
    env = ARCEnv(max_steps=15)
    vocab_size = 10
    d_model = 64
    action_space = env.action_space.n

    runner = TensorNeatRunner(
        model_class=DeltaNetARCModel, 
        model_kwargs=dict(vocab_size=vocab_size, d_model=d_model, action_space=action_space),
        population_size=args.pop_size
    )
    
    # Move entire population to the target device
    for model in runner.population:
        model.to(device)

    print(f"Starting evolution for {args.generations} generations with a population of {args.pop_size}...")
    
    # Optional progress bar over generations
    pbar = tqdm(range(1, args.generations + 1), desc="Training")
    for gen in pbar:
        fitness_scores = []
        
        # Evaluate each network population instance
        for idx, model in enumerate(runner.population):
            fit = evaluate_individual(model, env, num_episodes=args.episodes, device=device)
            fitness_scores.append(fit)
        
        max_fit = max(fitness_scores)
        avg_fit = sum(fitness_scores) / len(fitness_scores)
        
        # Update tqdm info
        pbar.set_postfix({"Max Fit": f"{max_fit:.4f}", "Avg Fit": f"{avg_fit:.4f}"})
        
        # Determine the best model for checkpoint tracking before reproducing
        best_idx = fitness_scores.index(max_fit)
        
        # Run Selection & Mutation Phase
        runner.select_and_reproduce(fitness_scores)
        
        # Periodically Save the best individual
        if gen % args.ckpt_freq == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"best_gen_{gen}.pt")
            # Pull the best individual from the pre-reproduced trackers (or just from index 0 since select_and_reproduce puts elites at the top)
            best_model_state = runner.population[0].state_dict()  
            torch.save(best_model_state, ckpt_path)

if __name__ == '__main__':
    main()
