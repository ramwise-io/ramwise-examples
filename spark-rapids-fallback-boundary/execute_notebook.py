"""Execute the generated notebook in place using nbclient."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path(__file__).parent / "spark_rapids_fallback_boundary.ipynb"
os.chdir(path.parent)
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=300, kernel_name="python3").execute()
nbformat.write(notebook, path)
print(path)
