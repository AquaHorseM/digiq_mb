import transformers
from digiq.environment import BatchedAndroidEnv
from digiq.models import AutoUIAgent
from digiq.algorithms import offpolicy_train_loop, eval_loop
from digiq.misc import colorful_print
from digiq.environment.android import EndResultEvaluator
from digiq.environment.android import autoui_translate_action
import wandb
import random
from omegaconf import DictConfig, OmegaConf
import os
import hydra
from accelerate.utils import set_seed
from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
# transformers.logging.set_verbosity_error()

def load_task_file(assets_path, task_set, task_split):
    all_tasks = []
    with open(os.path.join(assets_path, task_set + "_" + task_split + ".txt")) as fb: 
        for line in fb:
            all_tasks.append(line.replace("\n", ""))
    return all_tasks


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml
    (config), fg='red')
    try:
        from huggingface_hub import login
        login(token=config.huggingface_token)
    except:
        print(">>> Huggingface token not found.")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    initp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60*60))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, initp_kwargs], 
                              project_dir = config.save_path,
                              gradient_accumulation_steps=config.grad_accum_steps)
    
    # ensure each process launched by accelerator has a different seed
    # base seed should also be different for each process
    base_seed = random.randint(0, 1000)
    # Use accelerator.process_index (or local_process_index) to vary the seed across processes:
    set_seed(base_seed + accelerator.process_index)
    
    device = accelerator.device
    env = None
    if accelerator.is_main_process:
        # load environment
        all_tasks = load_task_file(config.assets_path, config.task_set, config.task_split)
        bsize = config.bsize
        base_port = 5554
        evaluators = [EndResultEvaluator(config.gemini_key, config.task_set)] * bsize
        assert len(evaluators) == bsize
        if config.agent_name == "autoui":
            translate_action = autoui_translate_action
            use_feature_extractor = True
    decode_f = lambda x:x
    if accelerator.is_main_process:
        colorful_print(">>> Agent: AutoUI", fg='blue')
        colorful_print(">>> Training algorithm: "+config.train_algorithm, fg='blue')
    
    if config.agent_name == "autoui":
        agent = AutoUIAgent(device=device, accelerator=accelerator, click_icon_path=config.click_icon_path,
                            temperature=config.temperature, do_sample=config.do_sample, 
                            policy_lm=config.policy_lm, critic_lm=config.critic_lm,
                            cache_dir=config.cache_dir, max_new_tokens=config.max_new_tokens,
                            learn_metric=config.learn_metric, 
                            advantage_estimation=config.advantage_estimation, api_endpoints=config.api_endpoints,
                            value_path=config.value_path, transition_path=config.transition_path)
        tokenizer = agent.tokenizer
    elif config.agent_name == "cogagent":
        agent = CogAgent(url=config.cogagent_url)
        tokenizer = None
    else:
        raise NotImplementedError("Only AutoUI agent is supported for now")

    if config.use_wandb and accelerator.is_main_process:
        wandb.login(key=config.wandb_key)
        wandb.init(project=config.project_name, entity=config.entity_name, name=config.run_name, config=dict(config))

    def construct_env(sample_mode):
        env = BatchedAndroidEnv(avd_name="test_Android", 
            cache_avd_names=[f"test{i}" for i in range(1,1+bsize)], 
            android_avd_home=config.android_avd_home,
            emulator_path=config.emulator_path, 
            adb_path=config.adb_path, 
            udids = [f"emulator-{base_port+2*i}" for i in range(bsize)],
            max_steps=config.max_steps-1, # will have 1 dangling step after stop signal is triggered
            appium_base_port = base_port+1098,
            run_headless=True, 
            use_feature_extractor=use_feature_extractor, 
            device=accelerator.device,
            translate_action=translate_action,
            evaluators=evaluators,
            temp_path = os.path.join(config.save_path, "images"),
            save_images=True,
            all_tasks=all_tasks,
            task_split=config.task_split,
            sample_mode=sample_mode,
            record=config.record,
        )
        return env

    if config.task_mode == "train_and_eval":
        offpolicy_train_loop(tokenizer=tokenizer,
                agent = agent,
                accelerator = accelerator,
                decode_f=decode_f,
                **config)
            
        if accelerator.is_main_process:
            env = construct_env(sample_mode=config.eval_sample_mode)
            eval_loop(env = env,
                    tokenizer=tokenizer,
                    agent = agent,
                    accelerator = accelerator,
                    decode_f=decode_f,
                    **config)
            
    elif config.task_mode == "train_and_remote_eval":
        offpolicy_train_loop(tokenizer=tokenizer,
                agent = agent,
                accelerator = accelerator,
                decode_f=decode_f,
                **config)
            
    elif config.task_mode == "eval":
        if accelerator.is_main_process:
            env = construct_env(sample_mode=config.eval_sample_mode)
            eval_loop(env = env,
                    tokenizer=tokenizer,
                    agent = agent,
                    accelerator = accelerator,
                    decode_f=decode_f,
                    **config)

if __name__ == "__main__":
    main()
