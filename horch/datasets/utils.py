import re

from torchvision.datasets.utils import check_integrity


def download_google_drive(url_or_id, root, filename, md5=None):
    match = re.match(
        r"https://drive.google.com/open\?id=(.*)", url_or_id)
    if match:
        file_id = match.group(1)
    else:
        file_id = url_or_id

    fpath = root / filename

    root.mkdir(exist_ok=True)

    # downloads file
    if fpath.is_file() and check_integrity(fpath, md5):
        print('Using downloaded and verified file: %s' % fpath)
    else:
        from google_drive_downloader import GoogleDriveDownloader as gdd
        gdd.download_file_from_google_drive(
            file_id=file_id,
            dest_path=fpath,
        )
