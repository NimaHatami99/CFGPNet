import logging
import os
import subprocess
import urllib
from pathlib import Path

import requests
import torch


def is_url(url, check=True):
    # Check if string is URL and check if URL exists
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (urllib.request.urlopen(url).getcode() == 200) if check else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False


def attempt_download(file, repo='ultralytics/yolov5', release='v7.0'):
    # Attempt file download from GitHub release assets if not found locally. release = 'latest', 'v7.0', etc.
    from utils.general import LOGGER

    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            file = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
            LOGGER.info(f'Found {url} locally at {file}')  # file already exists
            
            return file

        # GitHub assets
        assets = [f'yolov5{size}{suffix}.pt' for size in 'nsmlx' for suffix in ('', '6', '-cls', '-seg')]  # default
        try:
            tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
        except Exception:
            tag = release

        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        

    return str(file)
