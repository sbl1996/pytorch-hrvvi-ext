import os
import json
from pathlib import Path
from typing import Callable, Any


def read_lines(fp):
    return fmt_path(fp).read_text().splitlines()


def write_lines(fp, lines):
    fmt_path(fp).write_text(os.linesep.join(lines))
    return fp


def read_json(fp):
    with open(fp) as f:
        data = json.load(f)
    return data


def save_json(fp, obj):
    with open(fp, 'w') as f:
        json.dump(obj, f)


def fmt_path(fp):
    return Path(fp).expanduser().absolute()


def apply_dir(dir: Path, f: Callable[[Path], Any], suffix=None, recursive=True) -> None:
    if isinstance(dir, str):
        dir = fmt_path(dir)
    for fp in dir.iterdir():
        if fp.name.startswith('.'):
            continue
        elif fp.is_dir():
            if recursive:
                apply_dir(fp, f, recursive, suffix)
        elif fp.is_file():
            if suffix is None:
                f(fp)
            elif fp.suffix == suffix:
                f(fp)



def rename(fp: Path, new_name: str, stem=True):
    if stem:
        fp.rename(fp.parent / (new_name + fp.suffix))
    else:
        fp.rename(fp.parent / new_name)