import os
import argparse
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument("--input_image_dir", type=str, default="", help="Path to input images dir")
parser.add_argument("--output_image_dir", type=str, default="", help="Path to output smaller images dir")
args = parser.parse_args()

def resize_image(image_path, target_path):
    try:
        image = Image.open(image_path)
        # Resize the image to 0.25 of the original size
        image = image.resize((int(image.width * 0.25), int(image.height * 0.25)))
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # Create target directory if it doesn't exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        image.save(target_path)
        print(f"Resized {image_path} and saved to {target_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_directory(src_path, dst_path):
    for image_name in os.listdir(src_path):
        complete_src_path = os.path.join(src_path, image_name)
        complete_dst_path = os.path.join(dst_path, image_name)
        resize_image(complete_src_path, complete_dst_path)

if __name__ == "__main__":
    base_path = args.input_image_dir
    target_base_path = args.output_image_dir
    paths = [f"{base_path}/test{i}" for i in range(1, 9)]
    target_paths = [f"{target_base_path}/test{i}" for i in range(1, 9)]

    # Create a pool of 64 processes
    with ProcessPoolExecutor(max_workers=64) as executor:
        # Submit tasks to the pool
        futures = [executor.submit(process_directory, src, dst) for src, dst in zip(paths, target_paths)]
        # Ensure all tasks are completed
        for future in futures:
            future.result()
