from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = 'vineeth32/Machine-Failure-Prediction'
repo_type = 'dataset'

api = HfApi(token=os.getenv('HF_TOKEN'))

try:
  api.repo_info(repo_id=repo_id, repo_type=repo_type)
  print(f"Sapce '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
  print(f"Sapce '{repo_id}' not found. Creating new space...")
  create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
  print(f"Sapce '{repo_id}' created.")

api.upload_folder(
    folder_path='week_2_mls/data',
    repo_id = repo_id,
    repo_type = repo_type
)
