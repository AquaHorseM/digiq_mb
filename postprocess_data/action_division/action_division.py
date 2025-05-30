import argparse
import torch

# need to run in top
from digiq.environment.android import autoui_translate_action
from digiq.environment.android.autoui_utils import AndroidAction, ActionType

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def get_divided_actions(steps, tokenizer, model):
    subclasses = ["asking questions", "searching for things", "visiting websites"]
    results = []
    for i, step in tqdm(enumerate(steps), total=len(steps)):
        try:
            action = autoui_translate_action(step['action'])
        except:
            continue
        if action.action_type == ActionType.Type:
            prompt = f"A user is about to enter the following text in the input box on the screen of a phone: \"{action.typed_text}\". Select the option that most likely represents the user's intent among: {' '.join(subclasses)}. Only output the option content, without any additional information."

            messages = [
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            results.append((i, generated_text))
            print(f"{action.typed_text}, {results[-1]}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="Path to the LLM model")
    parser.add_argument("--input_data_path", type=str, default="", help="Path to collected trajectories (after reconstruction for RL)")
    parser.add_argument("--output_data_path", type=str, default="", help="Path to store data with divided action")
    args = parser.parse_args()

    input_data_path = args.input_data_path
    output_data_path = args.output_data_path

    # Load the dataset
    print("Loading dataset...")
    steps = torch.load(input_data_path, weights_only=False)
    print("Reconstructing dataset...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto")
    data = get_divided_actions(steps, tokenizer, model)

    # Save the dataset
    torch.save(data, output_data_path)