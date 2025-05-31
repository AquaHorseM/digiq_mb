import torch
import hydra
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta
import wandb

from digiq.algorithms.init_policy import get_initpolicy_trainer
from digiq.algorithms.train_policy_bremen import train_model_based_bremen
from digiq.models.encoder import GoalEncoder, ActionEncoder

class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.setup_accelerator()
        self.setup_encoders()
        
    def setup_accelerator(self):
        """Setup distributed training environment"""
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        initp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60*60))
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs, initp_kwargs],
            project_dir=self.config.train.save_path
        )
        
    def setup_encoders(self):
        """Initialize goal and action encoders"""
        self.goal_encoder = GoalEncoder(
            backbone=self.config.Goal_encoder.goal_encoder_backbone,
            cache_dir=self.config.Goal_encoder.goal_encoder_cache_dir,
            device=self.accelerator.device
        )
        
        self.action_encoder = ActionEncoder(
            backbone=self.config.Action_encoder.action_encoder_backbone,
            cache_dir=self.config.Action_encoder.action_encoder_cache_dir,
            device=self.accelerator.device
        )
        
    def initialize_policy(self):
        """Phase 1: Initialize policy using behavior cloning or MCP"""
        # Initialize wandb for tracking

            
        # Get the appropriate trainer
        trainer = get_initpolicy_trainer(
            self.config.train_init_policy.trainer_name,
            self.config,
            self.accelerator
        )
        
        # Run the initialization training
        trainer.train_loop(
            data_path=self.config.data.init_data_path,
            batch_size=self.config.data.batch_size,
            capacity=self.config.data.capacity,
            train_ratio=self.config.data.train_ratio,
            val_ratio=self.config.data.val_ratio
        )
        
        return trainer.agent
        
    def train_policy(self, policy):
        """Phase 2: RL training using Bremen style updates"""
        if self.accelerator.is_main_process:
            wandb.init(
                project=self.config.project_name,
                name=f"{self.config.run_name}_rl",
                config=dict(self.config)
            )
            
        # Train using Bremen style updates
        trained_policy = train_model_based_bremen(
            policy=policy,
            trans_model=policy.transition_model,  # Assuming policy has transition model
            num_iters=self.config.train_rl.num_iters,
            batch_size=self.config.train_rl.batch_size,
            rollout_length=self.config.train_rl.rollout_length,
            data_file=self.config.data.rl_data_path,
            gamma=self.config.train_rl.gamma,
            lam=self.config.train_rl.lam,
            clip_eps=self.config.train_rl.clip_eps,
            ent_coef=self.config.train_rl.ent_coef,
            bremen_epochs=self.config.train_rl.bremen_epochs,
            lr=self.config.train_rl.lr,
            goal_encoder=self.goal_encoder,
            action_encoder=self.action_encoder,
            device=self.accelerator.device
        )
        
        return trained_policy
        
    def run(self):
        """Run the complete training pipeline"""
        # Phase 1: Initialize policy
        print("Starting policy initialization...")
        policy = self.initialize_policy()
        
        # Phase 2: RL training
        print("Starting RL training...")
        final_policy = self.train_policy(policy)
        
        # Save the final policy
        if self.accelerator.is_main_process:
            torch.save(
                final_policy.state_dict(),
                f"{self.config.train.save_path}/final_policy.pth"
            )
            
        return final_policy

@hydra.main(config_name="training_pipeline", config_path="../../scripts/config/main", version_base="1.3")
def main(config):
    pipeline = TrainingPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
