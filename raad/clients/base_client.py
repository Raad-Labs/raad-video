# clients/base_client.py
import abc

class BaseVideoClient(abc.ABC):
    """
    Abstract base client to unify local/remote fetching.
    All clients must implement download() and exists() methods.
    """

    @abc.abstractmethod
    def exists(self, path_or_id: str) -> bool:
        pass

    @abc.abstractmethod
    def download(self, path_or_id: str, destination: str) -> None:
        """
        Download the file from the source to local destination.
        If local, this might be a copy operation.
        If remote, this might be an HTTP or streaming download.
        """
        pass

    @abc.abstractmethod
    def stream(self, path_or_id: str):
        """
        Potentially returns a file-like object or generator 
        for streaming the video data without downloading the entire file.
        """
        pass
