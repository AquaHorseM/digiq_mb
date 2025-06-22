from huggingface_hub import hf_hub_download, login
login()
file_path = hf_hub_download(
    repo_id="digiqmb/datasets",
    filename="general-ft.pt"
)