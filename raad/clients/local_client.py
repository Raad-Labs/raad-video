# clients/local_client.py
import os
import shutil
from pathlib import Path
from typing import BinaryIO, Generator, Optional

from .base_client import BaseVideoClient

class LocalClientError(Exception):
    """Base exception for local client errors."""
    pass

class LocalVideoClient(BaseVideoClient):
    """
    Accessing files directly on local disk or NAS paths.
    Provides robust error handling and efficient streaming.
    """
    def __init__(self, chunk_size: int = 8192):
        """Initialize with configurable chunk size for streaming."""
        self.chunk_size = chunk_size

    def exists(self, path_or_id: str) -> bool:
        """Check if file exists and is accessible."""
        try:
            path = Path(path_or_id)
            return path.exists() and path.is_file() and os.access(path, os.R_OK)
        except Exception as e:
            return False

    def download(self, path_or_id: str, destination: str) -> None:
        """
        Copy from source to destination with error handling.
        Creates parent directories if they don't exist.
        """
        try:
            src_path = Path(path_or_id)
            dst_path = Path(destination)

            if not self.exists(str(src_path)):
                raise LocalClientError(f"Source file does not exist or is not accessible: {path_or_id}")

            # Create parent directories if they don't exist
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy with metadata preservation
            shutil.copy2(src_path, dst_path)

        except (OSError, shutil.Error) as e:
            raise LocalClientError(f"Failed to copy {path_or_id} to {destination}: {str(e)}")

    def stream(self, path_or_id: str) -> Generator[bytes, None, None]:
        """
        Stream file contents in chunks.
        Uses context manager to ensure proper file handling.
        """
        if not self.exists(path_or_id):
            raise LocalClientError(f"File does not exist or is not accessible: {path_or_id}")

        try:
            with open(path_or_id, "rb") as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except IOError as e:
            raise LocalClientError(f"Error streaming file {path_or_id}: {str(e)}")

    def get_file_info(self, path_or_id: str) -> dict:
        """Get file metadata including size, modification time, etc."""
        try:
            path = Path(path_or_id)
            if not self.exists(str(path)):
                raise LocalClientError(f"File does not exist or is not accessible: {path_or_id}")

            stat = path.stat()
            return {
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "created": stat.st_ctime,
                "extension": path.suffix.lower(),
                "filename": path.name
            }
        except OSError as e:
            raise LocalClientError(f"Failed to get file info for {path_or_id}: {str(e)}")
