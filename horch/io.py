import os
import pickle
import sys
import stat
import json
import shutil
from pathlib import Path
from typing import Callable, Any, Union

PathLike = Union[str, Path]


def read_lines(fp: PathLike):
    return fmt_path(fp).read_text().splitlines()


def write_lines(lines: PathLike, fp):
    fmt_path(fp).write_text(os.linesep.join(lines))
    return fp


def read_pickle(fp: PathLike):
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(obj, fp: PathLike):
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)


def read_json(fp: PathLike):
    with open(fp) as f:
        data = json.load(f)
    return data


def save_json(obj, fp: PathLike):
    with open(fp, 'w') as f:
        json.dump(obj, f)


def fmt_path(fp: PathLike) -> Path:
    return Path(fp).expanduser().absolute()


def is_hidden(fp: PathLike):
    fp = fmt_path(fp)
    plat = sys.platform
    if plat == 'darwin':
        import Foundation
        url = Foundation.NSURL.fileURLWithPath_(str(fp))
        return url.getResourceValue_forKey_error_(None, Foundation.NSURLIsHiddenKey, None)[1]
    elif plat in ['win32', 'cygwin']:
        return bool(fp.stat().st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN)
    else:
        return fp.name.startswith(".")


def eglob(fp, pattern):
    fp = fmt_path(fp)
    for f in fp.glob(pattern):
        if not is_hidden(f):
            yield f


def apply_dir(dir: PathLike, f: Callable[[PathLike], Any], suffix=None, recursive=True) -> None:
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


def rename(fp: PathLike, new_name: str, stem=True):
    fp = fmt_path(fp)
    if stem:
        fp.rename(fp.parent / (new_name + fp.suffix))
    else:
        fp.rename(fp.parent / new_name)


def copy(src: PathLike, dst: PathLike):
    src = fmt_path(src)
    dst = fmt_path(dst)
    shutil.copy(str(src), str(dst))


def mv(src: PathLike, dst: PathLike):
    return fmt_path(shutil.move(str(src), str(dst)))


def rm(fp: PathLike):
    fp = fmt_path(fp)
    if fp.exists():
        if fp.is_dir():
            for d in fp.iterdir():
                rm(d)
            fp.rmdir()
        else:
            fp.unlink()