"""Single place for all file locations. Override with environment variables if needed:
LENSR_DATA (prepared datasets) and LENSR_ROOT (repository root)."""
import os

ROOT = os.environ.get("LENSR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("LENSR_DATA", os.path.join(ROOT, "data"))
GT = os.path.join(ROOT, "groundtruth")
RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "cache")
FIGURES = os.path.join(ROOT, "figures")
