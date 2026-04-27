from pathlib import Path
import json


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def figures_dir() -> Path:
    p = project_root() / "final_pic"
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir() -> Path:
    p = project_root() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
