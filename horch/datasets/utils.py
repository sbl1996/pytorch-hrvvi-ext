from torchvision.datasets.utils import check_integrity


def download_google_drive(file_id, root, filename, md5=None):
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
