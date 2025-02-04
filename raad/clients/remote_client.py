# clients/remote_client.py
import os
from pathlib import Path
from typing import Optional, Generator, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .base_client import BaseVideoClient

class RemoteClientError(Exception):
    """Base exception for remote client errors."""
    pass

class RemoteVideoClient(BaseVideoClient):
    """
    Implements fetching from a remote HTTP or custom service.
    Supports retries, timeouts, and proper error handling.
    Could also integrate with S3, GCS, or specialized data store.
    """
    def __init__(
        self,
        base_url: str,
        auth_token: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        chunk_size: int = 8192
    ):
        """
        Initialize with retry and timeout settings.
        
        Args:
            base_url: Base URL for the remote service
            auth_token: Optional authentication token
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            chunk_size: Size of chunks for streaming
        """
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.timeout = timeout
        self.chunk_size = chunk_size

        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with optional authentication."""
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def _get_url(self, path_or_id: str) -> str:
        """Construct full URL from path or ID."""
        return f"{self.base_url}/{path_or_id.lstrip('/')}"

    def exists(self, path_or_id: str) -> bool:
        """Check if remote resource exists using HEAD request."""
        try:
            response = self.session.head(
                self._get_url(path_or_id),
                headers=self._get_headers(),
                timeout=self.timeout
            )
            return response.status_code == 200
        except requests.RequestException as e:
            return False

    def download(self, path_or_id: str, destination: str) -> None:
        """
        Download file from remote source with progress tracking.
        Creates parent directories if they don't exist.
        """
        try:
            # Ensure parent directories exist
            Path(destination).parent.mkdir(parents=True, exist_ok=True)

            response = self.session.get(
                self._get_url(path_or_id),
                headers=self._get_headers(),
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

        except requests.RequestException as e:
            # Clean up partial download if it exists
            if os.path.exists(destination):
                os.remove(destination)
            raise RemoteClientError(f"Download failed for {path_or_id}: {str(e)}")

    def stream(self, path_or_id: str) -> Generator[bytes, None, None]:
        """
        Stream remote file in chunks with proper error handling.
        """
        try:
            response = self.session.get(
                self._get_url(path_or_id),
                headers=self._get_headers(),
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=self.chunk_size):
                if chunk:
                    yield chunk

        except requests.RequestException as e:
            raise RemoteClientError(f"Streaming failed for {path_or_id}: {str(e)}")

    def get_metadata(self, path_or_id: str) -> Dict[str, Any]:
        """Get metadata about the remote file."""
        try:
            response = self.session.head(
                self._get_url(path_or_id),
                headers=self._get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()

            return {
                "size": int(response.headers.get('content-length', 0)),
                "content_type": response.headers.get('content-type'),
                "last_modified": response.headers.get('last-modified'),
                "etag": response.headers.get('etag')
            }

        except requests.RequestException as e:
            raise RemoteClientError(f"Failed to get metadata for {path_or_id}: {str(e)}")
