from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path

# Load your Hugging Face token from a credentials file
def load_token(file_path="credentials.txt"):
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("HUGGINGFACE_TOKEN="):
                return line.strip().split("=")[1]
    raise ValueError("HUGGINGFACE_TOKEN not found in the file")


def load_and_save_model(model_name, hf_token, export_path):
    # Define export path
    export_path = Path(export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading the model")
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically map layers to devices
        token=hf_token,
        trust_remote_code=True,  # Trust custom code in the repository
    )
    print("[INFO] Model loaded successfully")

    print("\n[2/3] Loading the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )
    print("[INFO] Tokenizer loaded successfully.")

    print("\n[3/3] Saving the quantized model and tokenizer...")
    # Save the quantized model and tokenizer
    model.save_pretrained(export_path)
    tokenizer.save_pretrained(export_path)
    print(f"[INFO] model and tokenizer saved successfully to {export_path}")


if __name__ == "__main__":
    # Load the token
    print("\nLoading Hugging Face token...")
    HF_TOKEN = load_token()
    print("[INFO] Hugging Face token loaded successfully.")

    # Model and tokenizer name
    model_name = "meta-llama/Llama-3.1-8B"
    export_path = "llama_3.1_8B"

    print('Loading the model')
    load_and_save_model(model_name, HF_TOKEN, export_path)
    print("\n[INFO] Loading process completed successfully.")