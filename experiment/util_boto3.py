import boto3
import zipfile
import os
from typing import *
import util_network


def zip_and_upload(zipname: str, compfiles: Union[str, List[str]], timestamp: bool = False) -> str:
    """
    Save the given files (compfiles) into a zip file and uploads to our S3 bucket.
    The python files in the root is also saved in "archive" directory for the later debugging.
    :param zipname:
    :param compfiles:
    :return:
    """
    assert compfiles
    if timestamp:
        zipname = util_network.get_time_hash() + "_" + zipname
    with zipfile.ZipFile(zipname, "w", compression=zipfile.ZIP_DEFLATED) as new_zip:
        for pos, dirs, files in os.walk("."):
            if "archive" in pos:
                continue
            for file in files:
                _, ext = os.path.splitext(file)
                if ext == ".py":
                    new_zip.write(os.path.join(pos, file), arcname=os.path.join("archive", pos, file))
        if isinstance(compfiles, str):
            compfiles = [compfiles]
        for file in compfiles:
            basename = os.path.basename(file)
            new_zip.write(file, arcname=basename)

    s3 = boto3.resource("s3")
    bucket = s3.Bucket("tokudono-interpol")
    bucket.upload_file(zipname, zipname)
    return zipname


def download(zipname: str, targetdir: str=".") -> None:
    tgt = os.path.join(targetdir, zipname)
    if not os.path.exists(tgt):
        s3 = boto3.resource("s3")
        bucket = s3.Bucket("tokudono-interpol")
        bucket.download_file(zipname, tgt)


if __name__ == "__main__":
    zip_and_upload("test.zip", "README.md")
    download("pyonly.zip")
