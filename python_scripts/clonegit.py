import shutil
from pathlib import Path
import logging
from fastapi import HTTPException
from git import Repo, GitCommandError

logger = logging.getLogger(__name__)

REPO_FILE = Path(__file__).parent.parent / "repos.txt"

BASE_DIR = Path(__file__).parent.parent / "cloned"
BASE_DIR.mkdir(exist_ok=True)

def load_repos() -> set:
    if not REPO_FILE.exists():
        return set()
    with open(REPO_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())

def append_repo(url: str) -> None:
    with open(REPO_FILE, "a") as f:
        f.write(url + "\n")

def check_repos(url:str) -> Path:
    local_repos = load_repos()
    repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
    local_path = BASE_DIR / repo_name

    try:
        if url in local_repos:
            if local_path.exists():
                # Check if it's a valid git repository
                try:
                    repo = Repo(local_path)
                    logger.info(f"{url} already cloned. Pulling latest changes in {local_path}")
                    repo.remotes.origin.pull()
                except Exception:
                    # Not a valid git repo, remove and clone fresh
                    logger.warning(f"{local_path} exists but is not a valid git repo. Removing and cloning fresh.")
                    shutil.rmtree(local_path, ignore_errors=True)
                    Repo.clone_from(url, local_path, depth=1)
            else:
                logger.info(f"{url} is in repos.txt, but directory missing. Cloning into {local_path}")
                Repo.clone_from(url, local_path, depth=1)
        else:
            # URL not in repos.txt
            if local_path.exists():
                # Check if it's a valid git repository
                try:
                    repo = Repo(local_path)
                    logger.info(f"{url} directory exists and is a valid git repo. Using existing clone at {local_path}")
                    # Optionally pull latest changes
                    try:
                        repo.remotes.origin.pull()
                    except Exception as pull_err:
                        logger.warning(f"Failed to pull latest changes: {pull_err}. Using existing state.")
                except Exception:
                    # Not a valid git repo, remove and clone fresh
                    logger.warning(f"{local_path} exists but is not a valid git repo. Removing and cloning fresh.")
                    shutil.rmtree(local_path, ignore_errors=True)
                    Repo.clone_from(url, local_path, depth=1)
                    append_repo(url)
            else:
                logger.info(f"{url} is not in repos.txt. Cloning into {local_path}")
                Repo.clone_from(url, local_path, depth=1)
                append_repo(url)
    except GitCommandError as e:
        logger.exception(f"Git operation failed for {url}")
        raise HTTPException(status_code=400, detail=f"Failed to clone or pull repo: {e}")

    logger.info(f"Repo ready at {local_path}")
    return local_path

