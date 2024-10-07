import os
from huggingface_hub import HfApi, create_repo

def push_to_huggingface(repo_name, file_paths):
    # Initialize the Hugging Face API
    api = HfApi()

    # Create a new repository
    repo_url = create_repo(repo_name, private=False)
    print(f"Created new repository: {repo_url}")

    # Push each file to the repository
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_name,
            repo_id=repo_name,
        )
        print(f"Uploaded {file_name} to {repo_name}")

    print(f"All files have been pushed to {repo_url}")

# Example usage
if __name__ == "__main__":
    repo_name = "your-username/your-repo-name"  # Replace with your desired repo name
    file_paths = [
        "/path/to/your/model1.pt",
        "/path/to/your/model2.pt",
        # Add more file paths as needed
    ]
    
    push_to_huggingface(repo_name, file_paths)