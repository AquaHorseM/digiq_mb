from digiq.misc import colorful_print
import threading
import os
import torch
import time
import concurrent.futures
from tqdm import tqdm
import subprocess
import shutil
from glob import glob

def remote_eval_offline_ckpt(save_path, 
                            worker_temp_path, 
                            worker_run_path, 
                            worker_ips, 
                            worker_username):
    assert os.path.exists(os.path.join(save_path, "trainer_offline.pt")), "trainer_offline.pt not found"
    
    # add all workers into known hosts if not already
    colorful_print("Adding all workers to known hosts", fg='green')
    for worker_ip in worker_ips:
        print("worker_ip", worker_ip)
        os.system(f"ssh-keyscan -H {worker_ip} >> ~/.ssh/known_hosts")
    
    for worker_ip in worker_ips:
        os.system(f"ssh {worker_username}@{worker_ip} 'pkill -U {worker_username}'")
    time.sleep(10)
    for worker_ip in worker_ips:
        os.system(f"ssh {worker_username}@{worker_ip} 'skill -u {worker_username}'")
    time.sleep(10)

    # copy the agent to all remote workers
    colorful_print("Cleaning remote machine work temp path", fg='green')
    command = f"rm -rf {worker_temp_path} && mkdir -p {worker_temp_path} && exit"
    threads = []
    for worker_ip in worker_ips:
        t = threading.Thread(target=os.system, args=(f"""ssh -tt {worker_username}@{worker_ip} << EOF 
{command}
EOF
""",))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    def execute_scp(worker_ip):
        command = f"rsync -av --progress --exclude 'optimizer*' {save_path}/trainer_offline.pt/ {worker_username}@{worker_ip}:{worker_temp_path}/trainer_offline.pt/"
        os.system(command)

    # Colorful print (assuming the function is defined elsewhere)
    colorful_print("Copying the offline trainer to all workers", fg='green')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(execute_scp, worker_ips)

    colorful_print("Starting all trajectory collections", fg='green')
    threads = []
    command = f"""export CUDA_LAUNCH_BLOCKING=1 && conda activate digiq && cd {worker_run_path} && nohup python -u run.py --config-path config/main --config-name eval > evaluate.out 2>&1 & \n disown \n exit"""
    
    for worker_ip in worker_ips:
        t = threading.Thread(target=os.system, args=(f"""ssh -tt {worker_username}@{worker_ip} << EOF 
{command}
EOF
""",))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        colorful_print("Trajectory collection submitted", fg='green')

    # if global_step == offline_global_iterations - 1:
    #     colorful_print("Copying all trajs and images from all workers to this host machine", fg='green')
    #     for worker_ip in worker_ips:
    #         # remove existing images
    #         os.system(f"rm -rf {save_path}/images/{worker_ip}")
    #         os.system(f"scp -r {worker_username}@{worker_ip}:{worker_temp_path}/images {save_path}/images/{worker_ip}")
    #         os.system(f"scp {worker_username}@{worker_ip}:{worker_temp_path}/trajectories_eval.pt {save_path}/{worker_ip}")
    #         # the os.system is blocking, so we can directly load the trajs
        
    #     colorful_print("All trajs and images copied to this host machine", fg='green')

    #     # load all trajs in the remote machine
    #     trajectories_list = [torch.load(f"{save_path}/{worker_ip}") for worker_ip in worker_ips]
    #     trajectories = []
    #     for traj_list in trajectories_list:
    #         for traj in traj_list:
    #             trajectories.append(traj)
    #     # save all trajs
    #     torch.save(trajectories, os.path.join(save_path, "trajectories_eval.pt"))

    #     # merge images from all workers
    #     # merge: "{save_path}/images/{worker in worker_ip}/test1/*.png" -> "{save_path}/images/test1/*.png"
    #     # repeat for test2, test3, ..., test8
    #     # Loop through test folders
    #     for test_folder in range(1, 9):
    #         # Create the destination directory if it doesn't exist
    #         destination_dir = os.path.join(save_path, 'images', f'test{test_folder}')
    #         os.makedirs(destination_dir, exist_ok=True)
            
    #         # Loop through each worker and merge the images
    #         for worker in worker_ips:
    #             source_pattern = os.path.join(save_path, 'images', worker, f'test{test_folder}', '*.png')
    #             for file_path in glob(source_pattern):
    #                 # Copy the image to the destination directory
    #                 shutil.copy(file_path, destination_dir)

    #     colorful_print("All trajs and images merged", fg='green')

    return
