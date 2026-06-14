import os
import torch
import argparse
import neat
from tqdm import tqdm
from src.env.arc_env import ARCEnv
from src.agents.hyperneat_runner import HyperNEATDeltaNetBuilder
from src.scripts.train import evaluate_individual

def eval_genomes(genomes, config, builder, env, device, num_episodes):
    """
    Fitness evaluation function for neatly running populations
    """
    for genome_id, genome in genomes:
        # Build DeltaNet model from the CPPN genome
        model = builder.create_model(genome, config)
        
        # Evaluate model fitness
        fitness = evaluate_individual(model, env, num_episodes=num_episodes, device=device)
        genome.fitness = fitness

def main():
    parser = argparse.ArgumentParser(description="Kickoff Torcharc HyperNEAT Evolutionary Run")
    parser.add_argument("--generations", type=int, default=5000, help="Number of evolutionary iterations")
    parser.add_argument("--episodes", type=int, default=5, help="Number of ARC tasks evaluated per individual")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Directory to save model weights")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up Environment
    env = ARCEnv(max_steps=15)
    
    # Configure NEAT
    config_path = os.path.join(os.path.dirname(__file__), "neat-config.cfg")
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )
    
    # Track the population
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Save checkpoints periodically natively via NEAT-Python
    checkpoint_reporter = neat.Checkpointer(generation_interval=50, filename_prefix=f"{args.ckpt_dir}/hyperneat-chkpt-")
    p.add_reporter(checkpoint_reporter)

    builder = HyperNEATDeltaNetBuilder(
        config=config, 
        vocab_size=10, 
        d_model=64, 
        dim_head=64, 
        action_space=env.action_space.n, 
        device=device
    )

    print(f"Starting HyperNEAT evolution for {args.generations} generations...")
    
    # Wrap eval function with arguments
    def eval_wrapper(genomes, config):
        eval_genomes(genomes, config, builder, env, device, args.episodes)
        
    winner = p.run(eval_wrapper, args.generations)
    
    print("\nBest genome found throughout evolution:")
    print(winner)

if __name__ == '__main__':
    main()
