import argparse
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

def test_llms(model, steps, actions):
    for action in actions:
        idx, subclass = action
        prompt = f"A phone user wants to acheive the following goal: \"{steps[idx]['task']}\". The user needs to enter text in the input box of the user interface at a single step: **{subclass}**. Provide the text the user should input. Your response should contain only the text the user should enter, without any additional information."
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
        print(prompt)
        print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="Path to the LLM model")
    parser.add_argument("--input_data_path", type=str, default="", help="Path to collected trajectories (after reconstruction for RL)")
    parser.add_argument("--action_data_path", type=str, default="", help="Path to store data with divided action")
    args = parser.parse_args()

    input_data_path = args.input_data_path
    action_data_path = args.action_data_path

    # Load the dataset
    print("Loading dataset...")
    steps = torch.load(input_data_path, weights_only=False)
    actions = torch.load(action_data_path, weights_only=False)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto")
    test_llms(model, steps, actions)