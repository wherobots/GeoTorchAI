import os
import urllib.request
import gzip
import tarfile
import zipfile
import torch
from torch.utils.model_zoo import tqdm
from .exceptions import FileDownloadException, ExtractArchiveException


def _download_remote_file(file_url: str, save_path: str) -> None:
	os.makedirs(save_path, exist_ok=True)
	file_name  = file_url.split("/")[-1]
	full_save_path = os.path.join(save_path, file_name)

	for _ in range(4):
		with urllib.request.urlopen(urllib.request.Request(file_url, headers = {"Method": "HEAD", "User-Agent": "pytorch/geotorch"})) as response:
			if response.url == file_url or response.url is None:
				break
			file_url = response.url
	else:
		raise FileDownloadException("Unable to download the requested file")

	print("File downloading started...")
	with urllib.request.urlopen(urllib.request.Request(file_url, headers={"User-Agent": "pytorch/geotorch"})) as response:
		_save_chunk(iter(lambda: response.read(1024 * 32), b""), full_save_path, response.length)
	print("File downloading finished")



def _extract_archive(from_path, to_path) -> None:
	if _is_tar(from_path):
		print("Archive extraction started...")
		with tarfile.open(from_path, 'r') as tar:
			tar.extractall(path=to_path)
		print("Archive extraction finished")
	elif _is_targz(from_path):
		print("Archive extraction started...")
		with tarfile.open(from_path, 'r:gz') as tar:
			tar.extractall(path=to_path)
		print("Archive extraction finished")
	elif _is_gzip(from_path):
		print("Archive extraction started...")
		to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
		with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
			out_f.write(zip_f.read())
		print("Archive extraction finished")
	elif _is_zip(from_path):
		print("Archive extraction started...")
		with zipfile.ZipFile(from_path, 'r') as z:
			z.extractall(to_path)
		print("Archive extraction finished")
	else:
		raise ExtractArchiveException("Extraction of {} not supported".format(from_path))



def _save_chunk(chunks, save_path, response_size):
	with open(save_path, "wb") as f, tqdm(total=response_size) as progress:
		for chunk in chunks:
			if not chunk:
				continue
			f.write(chunk)
			progress.update(len(chunk))



def _is_tar(filename):
	return filename.endswith(".tar")


def _is_targz(filename):
	return filename.endswith(".tar.gz")


def _is_gzip(filename):
	return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
	return filename.endswith(".zip")




