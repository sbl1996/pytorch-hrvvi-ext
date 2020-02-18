import os
import pickle
import sys
import stat
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


def read_pickle(fp):
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(obj, fp):
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)


def save_json(fp, obj):
    with open(fp, 'w') as f:
        json.dump(obj, f)


def fmt_path(fp):
    return Path(fp).expanduser().absolute()


def is_hidden(fp):
    fp = fmt_path(fp)
    if sys.platform == 'darwin':
        import Foundation
        url = Foundation.NSURL.fileURLWithPath_(str(fp))
        return url.getResourceValue_forKey_error_(None, Foundation.NSURLIsHiddenKey, None)[1]
    else:
        return bool(fp.stat().st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN)


def eglob(fp, pattern):
    fp = fmt_path(fp)
    for f in fp.glob(pattern):
        if not is_hidden(f):
            yield f


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