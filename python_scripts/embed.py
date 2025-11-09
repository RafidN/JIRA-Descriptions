# python_scripts/embed.py

from __future__ import annotations

import os
import re
import time
import math
import errno
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

logger = logging.getLogger(__name__)

# Constants --------------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")
EMBED_DIM: int | None = None  # will be set after first embedding
MAX_FILE_BYTES = 300_000
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 150

MAX_EMBED_CHUNKS = int(os.getenv("MAX_EMBED_CHUNKS", "2500"))  # cap to speed up
EMBED_TIMEOUT_SEC = int(os.getenv("EMBED_TIMEOUT_SEC", "90"))  # abort if too long
FILTER_DIR_PREFIX = os.getenv("FILTER_DIR_PREFIX")  # e.g. "fastapi" to only embed that subdir

# Cache directory - relative to workspace root
CACHE_DIR = Path(".embeddings_cache")

TEXT_EXTS = {
    ".py",".js",".jsx",".ts",".tsx",".json",".yml",".yaml",".toml",".md",".txt",
    ".css",".scss",".less",".html",".htm",".java",".kt",".kts",".rb",".go",".rs",
    ".c",".h",".cpp",".hpp",".cc",".m",".mm",".swift",".cs",".fs",".php",".sh",
    ".bash",".zsh",".sql",".ini",".cfg",".pl",".pm",".r",".jl",".gitignore",
    ".gitattributes",".editorconfig"
}

SKIP_DIR_PATTERNS = [
    r"\.git($|/)", r"node_modules($|/)", r"dist($|/)", r"build($|/)",
    r"\.venv($|/)", r"venv($|/)", r"__pycache__($|/)", r"\.pytest_cache($|/)",
    r"\.mypy_cache($|/)", r"target($|/)", r"out($|/)", r"bin($|/)", r"obj($|/)"
]

SKIP_FILE_PATTERNS = [
    r"\.lock$", r"\.min\.js$", r"\.min\.css$", r"\.map$",
    r"\.(png|jpg|jpeg|gif|svg|webp|ico)$", r"\.(pdf|zip|tar|gz|bz2|7z)$",
    r"\.wasm$", r"\.(ttf|otf|woff2?)$", r"package-lock\.json$",
    r"pnpm-lock\.yaml$", r"yarn\.lock$"
]

_GENAI_READY = False

def _ensure_genai_configured():
    global _GENAI_READY
    if _GENAI_READY:
        return
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv, find_dotenv  # type: ignore
            load_dotenv(find_dotenv(usecwd=True) or ".env")
            api_key = os.getenv("GOOGLE_API_KEY")
        except Exception:
            pass
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    genai.configure(api_key=api_key)
    _GENAI_READY = True
    logger.info("GenAI configured")

# --- Data structures ---------------------------------------------------------

@dataclass
class Chunk:
    path: Path
    idx: int
    text: str

# --- Helpers ----------------------------------------------------------------

def _is_probably_text_file(path: Path) -> bool:
    if path.name.lower() in {"dockerfile","makefile"}:
        return True
    return path.suffix.lower() in TEXT_EXTS

def _should_skip_file(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    for pat in SKIP_FILE_PATTERNS:
        if re.search(pat, rel):
            return True
    if path.name == ".env":
        return True
    if not _is_probably_text_file(path):
        return True
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return True
    except OSError:
        return True
    return False

def _iter_candidate_files(root: Path) -> Iterable[Path]:
    skip_dir_regexes = [re.compile(pat) for pat in SKIP_DIR_PATTERNS]
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix() + "/"
        dirnames[:] = [
            d for d in dirnames
            if not any(r.search(f"{rel_dir}{d}/") for r in skip_dir_regexes)
        ]
        for fname in filenames:
            p = Path(dirpath) / fname
            if _should_skip_file(p, root):
                continue
            yield p

def _read_text(path: Path) -> str:
    try:
        data = path.read_bytes()
        if len(data) == 0 or b"\x00" in data:
            return ""
        if len(data) > MAX_FILE_BYTES:
            data = data[:MAX_FILE_BYTES]
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[Tuple[int,int,str]]:
    if chunk_size <= 0:
        chunk_size = DEFAULT_CHUNK_SIZE
    if overlap < 0:
        overlap = 0
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        seg = text[i:j]
        if seg.strip():
            out.append((i,j,seg))
        if j == n:
            break
        i = j - overlap if overlap > 0 else j
        if i <= 0:
            i = j
    return out

def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def _get_cache_key(text: str, model: str = EMBED_MODEL) -> str:
    """Generate a cache key based on text content and model."""
    content = f"{model}:{text}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def _get_cache_path(cache_key: str) -> Path:
    """Get the cache file path for a given cache key."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{cache_key}.npy"

def _load_from_cache(cache_key: str) -> np.ndarray | None:
    """Load embedding from cache if it exists."""
    cache_path = _get_cache_path(cache_key)
    if cache_path.exists():
        try:
            embedding = np.load(cache_path)
            logger.debug(f"Loaded embedding from cache: {cache_key[:16]}...")
            # Ensure proper shape (1, dim) for consistency
            if embedding.ndim == 1:
                return embedding.reshape(1, -1)
            elif embedding.ndim == 2:
                return embedding
            else:
                logger.warning(f"Unexpected embedding shape in cache: {embedding.shape}, regenerating")
                return None
        except Exception as e:
            logger.warning(f"Failed to load cache for {cache_key[:16]}...: {e}")
            return None
    return None

def _save_to_cache(cache_key: str, embedding: np.ndarray) -> None:
    """Save embedding to cache."""
    try:
        cache_path = _get_cache_path(cache_key)
        # Ensure it's a 1D array for storage (will reshape on load)
        to_save = embedding.flatten() if embedding.ndim > 1 else embedding
        np.save(cache_path, to_save)
        logger.debug(f"Saved embedding to cache: {cache_key[:16]}...")
    except Exception as e:
        logger.warning(f"Failed to save cache for {cache_key[:16]}...: {e}")

def _embed_one(text: str, retries: int = 3, backoff: float = 0.8) -> np.ndarray:
    """Embed text, checking cache first and saving to cache after generation."""
    _ensure_genai_configured()
    global EMBED_DIM
    
    # Check cache first
    cache_key = _get_cache_key(text, EMBED_MODEL)
    cached = _load_from_cache(cache_key)
    if cached is not None:
        if EMBED_DIM is None:
            EMBED_DIM = cached.shape[1]
            logger.info(f"Detected embedding dimension EMBED_DIM={EMBED_DIM} (from cache)")
        elif cached.shape[1] == EMBED_DIM:
            return cached
        else:
            # Dimension mismatch - remove bad cache and regenerate
            logger.warning(f"Cached embedding dimension mismatch {cached.shape[1]} != {EMBED_DIM}, removing cache and regenerating")
            try:
                cache_path = _get_cache_path(cache_key)
                if cache_path.exists():
                    cache_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove bad cache file: {e}")
    
    # Generate embedding if not in cache
    last = None
    for attempt in range(retries):
        try:
            r = genai.embed_content(model=EMBED_MODEL, content=text)
            vec = np.array(r["embedding"], dtype=np.float32).reshape(1, -1)
            if EMBED_DIM is None:
                EMBED_DIM = vec.shape[1]
                logger.info(f"Detected embedding dimension EMBED_DIM={EMBED_DIM}")
            elif vec.shape[1] != EMBED_DIM:
                raise RuntimeError(f"Inconsistent embedding dim {vec.shape[1]} != {EMBED_DIM}")
            
            # Save to cache
            _save_to_cache(cache_key, vec)
            return vec
        except Exception as e:
            last = e
            time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"Embedding failed: {last}")

def _language_from_extension(path: Path) -> str:
    return {
        ".py":"python",".js":"javascript",".jsx":"javascript",".ts":"typescript",".tsx":"tsx",
        ".json":"json",".yml":"yaml",".yaml":"yaml",".toml":"toml",".md":"markdown",".css":"css",
        ".scss":"scss",".less":"less",".html":"html",".htm":"html",".java":"java",".kt":"kotlin",
        ".kts":"kotlin",".rb":"ruby",".go":"go",".rs":"rust",".c":"c",".h":"c",".cpp":"cpp",
        ".hpp":"cpp",".cc":"cpp",".m":"objectivec",".mm":"objectivec",".swift":"swift",".cs":"csharp",
        ".php":"php",".sh":"bash",".bash":"bash",".zsh":"bash",".sql":"sql",".ini":"ini",".cfg":"ini",
        ".pl":"perl",".pm":"perl",".r":"r",".jl":"julia"
    }.get(path.suffix.lower(),"")

# --- Public API --------------------------------------------------------------

def build_context_for_ticket(
    repo_dir: Path,
    query: str,
    max_k: int = 12,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> Tuple[str,List[str],int]:
    global EMBED_DIM
    if not isinstance(repo_dir, Path):
        repo_dir = Path(repo_dir)
    logger.info(f"Indexing repo={repo_dir} query='{query[:60]}'")
    start_time = time.time()

    try:
        files = list(_iter_candidate_files(repo_dir))
    except Exception as e:
        logger.exception("Failed enumerating files")
        raise
    if FILTER_DIR_PREFIX:
        files = [f for f in files if f.relative_to(repo_dir).as_posix().startswith(FILTER_DIR_PREFIX)]
        logger.info(f"Filtered by prefix '{FILTER_DIR_PREFIX}' -> files={len(files)}")
    logger.info(f"Candidate files={len(files)}")
    if not files:
        return "", [], 0

    chunk_texts: List[str] = []
    meta: List[Dict] = []
    for p in files:
        txt = _read_text(p)
        if not txt.strip():
            continue
        for ci,(s,e,c) in enumerate(_chunk_text(txt, chunk_size, overlap)):
            meta.append({"path":p,"start":s,"end":e,"chunk_idx":ci})
            chunk_texts.append(c)
    total_chunks = len(chunk_texts)
    logger.info(f"Total chunks={total_chunks}")

    if total_chunks == 0:
        return "", [], 0

    # Cap chunks
    if total_chunks > MAX_EMBED_CHUNKS:
        logger.info(f"Capping chunks {total_chunks} -> {MAX_EMBED_CHUNKS}")
        meta = meta[:MAX_EMBED_CHUNKS]
        chunk_texts = chunk_texts[:MAX_EMBED_CHUNKS]
        total_chunks = len(chunk_texts)

    logger.info("Embedding chunks...")
    embeds = []
    per_chunk_times: List[float] = []
    cache_hits = 0
    cache_misses = 0
    for i, t in enumerate(chunk_texts, start=1):
        t0 = time.time()
        # Check cache before calling _embed_one
        cache_key = _get_cache_key(t, EMBED_MODEL)
        cached_vec = _load_from_cache(cache_key)
        was_cached = False
        
        if cached_vec is not None:
            # Validate cached embedding dimension
            if EMBED_DIM is None:
                EMBED_DIM = cached_vec.shape[1]
                logger.info(f"Detected embedding dimension EMBED_DIM={EMBED_DIM} (from cache)")
                was_cached = True
            elif cached_vec.shape[1] == EMBED_DIM:
                was_cached = True
            else:
                # Dimension mismatch - invalidate cache
                logger.warning(f"Cached embedding dimension mismatch {cached_vec.shape[1]} != {EMBED_DIM}, regenerating")
                try:
                    cache_path = _get_cache_path(cache_key)
                    if cache_path.exists():
                        cache_path.unlink()
                except Exception:
                    pass
                cached_vec = None
        
        if was_cached and cached_vec is not None:
            cache_hits += 1
            embeds.append(cached_vec)
        else:
            cache_misses += 1
            embeds.append(_embed_one(t))
        
        dt = time.time() - t0
        per_chunk_times.append(dt)
        if i % 100 == 0 or i in (1, total_chunks):
            avg = sum(per_chunk_times)/len(per_chunk_times)
            # Estimate ETA: average time per item * remaining items (assuming similar cache hit rate)
            remaining = total_chunks - i
            eta = avg * remaining
            cache_pct = (cache_hits / i * 100) if i > 0 else 0
            logger.info(f"Embedded {i}/{total_chunks} avg={avg:.3f}s ETA={eta:.1f}s cache_hits={cache_hits} ({cache_pct:.1f}%)")
        if time.time() - start_time > EMBED_TIMEOUT_SEC:
            logger.warning(f"Embedding timeout after {i} chunks; continuing with partial set")
            break
    if total_chunks > 0:
        final_cache_pct = (cache_hits / len(embeds) * 100) if embeds else 0
        logger.info(f"Cache statistics: {cache_hits} hits, {cache_misses} misses ({final_cache_pct:.1f}% hit rate)")
    # If timeout triggered, slice embeds/meta/chunk_texts
    embedded_count = len(embeds)
    meta = meta[:embedded_count]
    chunk_texts = chunk_texts[:embedded_count]

    if embedded_count == 0:
        return "", [], 0

    # Ensure EMBED_DIM is set from the embeddings we have
    if EMBED_DIM is None:
        # Get dimension from first embedding
        if embeds and len(embeds) > 0:
            EMBED_DIM = embeds[0].shape[1]
            logger.info(f"Detected embedding dimension EMBED_DIM={EMBED_DIM} from collected embeddings")
        else:
            raise RuntimeError("Embedding dimension not initialized and no embeddings available")

    X = np.vstack(embeds).astype(np.float32)
    X = _normalize_rows(X)

    logger.info(f"Building FAISS index dim={EMBED_DIM} vectors={embedded_count}")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(X)

    logger.info("Embedding query / retrieving")
    q = _normalize_rows(_embed_one(query))
    k = int(max(1, min(max_k, embedded_count)))
    sims, idxs = index.search(q, k)
    picked = idxs[0].tolist()
    logger.info(f"Retrieved top_k={k}")

    used_files: List[str] = []
    parts: List[str] = ["## Retrieved Repository Context (Top-K)\n"]
    for rank, idx in enumerate(picked, start=1):
        m = meta[idx]
        path_rel = m["path"].relative_to(repo_dir).as_posix()
        if path_rel not in used_files:
            used_files.append(path_rel)
        lang = _language_from_extension(m["path"])
        header = f"### {rank}. {path_rel} (chunk {m['chunk_idx']}, {m['start']}–{m['end']})"
        parts.append(header)
        body = chunk_texts[idx].rstrip()
        fenced = f"```{lang}\n{body}\n```" if lang else f"```\n{body}\n```"
        parts.append(fenced)

    context_md = "\n".join(parts)
    logger.info(f"Context ready used_chunks={len(picked)} unique_files={len(used_files)} total_time={time.time()-start_time:.2f}s")
    return context_md, used_files, len(picked)

__all__ = ["build_context_for_ticket"]
