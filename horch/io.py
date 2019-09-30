import json
from pathlib import Path
from typing import Callable, Any


def read_json(fp):
    with open(fp) as f:
        data = json.load(f)
    return data


def save_json(fp, obj):
    with open(fp, 'w') as f:
        json.dump(obj, f)


def fmt_path(fp):
    return Path(fp).expanduser().absolute()


def apply_dir(dir: Path, f: Callable[[Path], Any], recursive=True) -> None:
    for fp in dir.iterdir():
        # hidden files
        if fp.name.startswith('.'):
            continue
        elif fp.is_dir():
            if recursive:
                apply_dir(fp, f, recursive)
            continue
        elif fp.is_file():
            f(fp)
        else:
            continue


def rename(fp: Path, new_name: str, stem=True):
    if stem:
        fp.rename(fp.parent / (new_name + fp.suffix))
    else:
        fp.rename(fp.parent / new_name)