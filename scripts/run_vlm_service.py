from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from pydantic import BaseModel
import uvicorn
import torch
from io import BytesIO

from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from PIL import Image

prompt = "You're given a user interface. List all clickable items' positions. Positions are defined as coordinates in the screen which need to be normalized to 0 to 1, e.g. (0.8, 0.2)."

# Assume you have your own LLM inference function here:
# from my_llm_inference import load_model, generate_text

def generate_text(model, prompt: str, max_tokens: int = 50):
    # Replace this stub with your actual generate logic
    # Example: return generate_text(model, prompt, max_tokens)
    return f"Generated response for prompt: {prompt}"  

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

class GenerateResponse(BaseModel):
    generated_text: str

app = FastAPI(title="Local LLM Inference Service")

@app.on_event("startup")
def startup_event():
    global processor, model

    model_path = '/data/mqj/models/qwen2.5-vl-7b-finetuned'

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(model_path).to("cuda:1")

    if model is None:
        raise RuntimeError(f"Failed to load model from {model_path}")

@app.post("/state")
async def get_state_representation(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda:1")

        with torch.no_grad():
            outputs = model(
                **inputs,
                return_dict=True,
                output_hidden_states=True
            )
            last_hidden_state = outputs.hidden_states[-1]
            rep = last_hidden_state[:, -1, :][0]
        
        buf = BytesIO()
        torch.save(rep.cpu(), buf)
        buf.seek(0)
        return Response(content=buf.read(), media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)