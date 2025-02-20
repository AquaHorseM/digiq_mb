filepath = "./llava_launched_places.out"

# example: Running on public URL: https://81208f316c4b7ec253.gradio.live
import re
import os
import sys

capacity = 16

counter = 0
with open(filepath, "r") as f:
    for line in f:
        if "Running on public URL" in line:
            url = re.findall(r"https://\w+\.gradio\.live", line)[0]
            print(f'"{url}",')
            counter += 1
            
print(f"Total number of URLs: {counter}")
        # if counter == capacity:
        #     print(f"Capacity reached: {capacity}")
        #     counter = 0
            