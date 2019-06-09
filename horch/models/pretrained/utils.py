from pathlib import Path

import torch
from horch.datasets.utils import download_google_drive


def load_state_dict_from_google_drive(file_id, filename, md5, model_dir=None, map_location=None):
    r"""Loads the Torch serialized object at the given URL.

    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.

    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if not set.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr

    """

    if model_dir is None:
        from torch.hub import _get_torch_home
        torch_home = _get_torch_home()
        torch_home = Path(torch_home)
        model_dir = torch_home / 'checkpoints'
    else:
        model_dir = Path(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    download_google_drive(file_id, model_dir, filename, md5)
    cached_file = model_dir / filename
    return torch.load(cached_file, map_location=map_location)
