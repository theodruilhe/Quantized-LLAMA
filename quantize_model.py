from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load your Hugging Face token from a credentials file
def load_token(file_path="credentials.txt"):
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("HUGGINGFACE_TOKEN="):
                return line.strip().split("=")[1]
    raise ValueError("HUGGINGFACE_TOKEN not found in the file")

def quantize_and_save_model(model_name, hf_token, export_path):
    # Define export path
    export_path = Path(export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading the model with 8-bit quantization...")
    # Configure bitsandbytes quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 8-bit quantization
    )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",  # Automatically map layers to devices
        token=hf_token,
        trust_remote_code=True,  # Trust custom code in the repository
    )
    print("[INFO] Model loaded successfully with 8-bit quantization.")

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
    print(f"[INFO] Quantized model and tokenizer saved successfully to {export_path}")

if __name__ == "__main__":
    # Load the token
    print("\nLoading Hugging Face token...")
    HF_TOKEN = load_token()
    print("[INFO] Hugging Face token loaded successfully.")

    # Model and tokenizer name
    model_name = "meta-llama/Llama-3.1-8B"
    export_path = "./quantized_model"

    print("\nStarting the 8-bit quantization process using bitsandbytes...")
    quantize_and_save_model(model_name, HF_TOKEN, export_path)
    print("\n[INFO] 8-bit quantization process completed successfully.")