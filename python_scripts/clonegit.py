import shutil
import tempfile
from pathlib import Path
import logging
from fastapi import HTTPException
from git import Repo

logger = logging.getLogger(__name__)

def shallow_clone(url: str) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="repo_"))
    logger.info(f"Cloning {url} into {tmpdir} ...")
    try:
        Repo.clone_from(url, tmpdir, depth=1)
    except Exception as e:
        logger.exception(f"Failed to clone {url}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed to clone repo: {e}")
    logger.info(f"Clone complete: {tmpdir}")
    return tmpdir

def cleanup_repo(repo_dir: Path) -> None:
    logger.info(f"Removing temp repo {repo_dir}")
    shutil.rmtree(repo_dir, ignore_errors=True)
