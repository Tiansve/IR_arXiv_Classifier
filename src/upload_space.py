"""Step 5c — push the Gradio app to a Hugging Face Space.

Creates (if needed) ``HF_SPACE_REPO`` with sdk=gradio and uploads everything
under ``space/`` (app.py, requirements.txt, README.md, examples.json).

Prerequisite::

    huggingface-cli login

CLI: ``python -m src.upload_space``
"""
from __future__ import annotations

import logging

from huggingface_hub import HfApi, create_repo

from .config import HF_SPACE_REPO, ROOT
from .utils import setup_logging

log = logging.getLogger("upload_space")

SPACE_DIR = ROOT / "space"
FILES = ["app.py", "requirements.txt", "README.md", "examples.json"]


def main() -> None:
    setup_logging()

    log.info("Ensuring space repo exists: %s", HF_SPACE_REPO)
    create_repo(
        HF_SPACE_REPO,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=False,
    )

    api = HfApi()
    for name in FILES:
        path = SPACE_DIR / name
        if not path.exists():
            log.warning("Skip missing %s", path)
            continue
        log.info("Uploading %s", name)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=HF_SPACE_REPO,
            repo_type="space",
        )

    url = f"https://huggingface.co/spaces/{HF_SPACE_REPO}"
    print(f"\nSpace live at: {url}  (build may take 1-3 minutes)\n")


if __name__ == "__main__":
    main()
