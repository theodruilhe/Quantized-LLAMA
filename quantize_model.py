from transformers import AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.exporters.onnx import export
from optimum.exporters.tasks import TasksManager
from pathlib import Path
import torch
import warnings
from tqdm import tqdm  # For progress bars

# Suppress TracerWarnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

# Load your Hugging Face token from a credentials file
def load_token(file_path="credentials.txt"):
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("HUGGINGFACE_TOKEN="):
                return line.strip().split("=")[1]
    raise ValueError("HUGGINGFACE_TOKEN not found in the file")

def quantize_and_save_model(model_name, hf_token, export_path, opset=14):
    # Define export paths
    export_path = Path(export_path)
    onnx_file_path = export_path / "model.onnx"  # ONNX model file path
    params_export_path = export_path / "parameters"  # Parameters folder path
    params_export_path.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading the model...")
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
    print("[INFO] Model loaded successfully.")

    print("\n[2/4] Creating ONNX configuration...")
    # Create ONNX configuration
    onnx_config = TasksManager.get_exporter_config_constructor(
        exporter="onnx",
        model=model,
        task="text-generation",
        library_name="transformers"
    )(model.config)

    # Set `use_cache` to False
    onnx_config.use_cache = False
    print("[INFO] ONNX configuration created.")

    print("\n[3/4] Exporting the model to ONNX format...")
    # Export the model to ONNX
    export(
        model=model,
        config=onnx_config,
        output=onnx_file_path,
        opset=opset
    )
    print(f"[INFO] ONNX model exported successfully to {onnx_file_path}")

    print("\n[4/4] Applying 8-bit quantization...")
    # Define the quantization configuration for 8-bit quantization
    quantization_config = AutoQuantizationConfig.arm64(is_static=False)  # Use dynamic quantization

    # Load the ONNX model with quantization
    quantized_model = ORTModelForCausalLM.from_pretrained(
        export_path,
        quantization_config=quantization_config,
        provider="CPUExecutionProvider"
    )

    # Save all quantized model parameters
    print("\nSaving quantized model parameters to the dedicated folder...")
    for param_name, param_tensor in tqdm(quantized_model.state_dict().items(), desc="Saving parameters", unit="param"):
        param_file_path = params_export_path / f"{param_name.replace('.', '_')}.pth"
        torch.save(param_tensor, param_file_path)

    print(f"\n[INFO] 8-bit Quantized model and parameters saved successfully to {export_path}")

if __name__ == "__main__":
    # Load the token
    print("\nLoading Hugging Face token...")
    HF_TOKEN = load_token()
    print("[INFO] Hugging Face token loaded successfully.")

    # Model and tokenizer name
    model_name = "meta-llama/Llama-3.2-1B"
    export_path = "./quantized_model"

    print("\nStarting the 8-bit quantization and export process...")
    quantize_and_save_model(model_name, HF_TOKEN, export_path)
    print("\n[INFO] 8-bit quantization process completed successfully.")