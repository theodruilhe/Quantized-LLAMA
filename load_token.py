def load_token(file_path="credentials.txt"):
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("HUGGINGFACE_TOKEN="):
                return line.strip().split("=")[1]
    raise ValueError("HUGGINGFACE_TOKEN not found in the file")
