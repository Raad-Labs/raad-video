# clients/base_client.py
import abc
from typing import AsyncIterator, Optional, Union, BinaryIO
from enum import Enum

class TransportProtocol(Enum):
    HTTP = "http"
    QUIC = "quic"
    WEBTRANSPORT = "webtransport"

class BaseVideoClient(abc.ABC):
    """
    Abstract base client to unify local/remote fetching with support for
    modern transport protocols and efficient frame streaming.
    """

    def __init__(self, transport: TransportProtocol = TransportProtocol.QUIC):
        self.transport = transport

    @abc.abstractmethod
    async def exists(self, path_or_id: str) -> bool:
        """Check if resource exists."""
        pass

    @abc.abstractmethod
    async def download(self, path_or_id: str, destination: str) -> None:
        """Download file with support for modern transport protocols."""
        pass

    @abc.abstractmethod
    async def stream(self, path_or_id: str, batch_size: int = 32) -> AsyncIterator[bytes]:
        """Stream video frames with configurable batching."""
        pass

    @abc.abstractmethod
    async def stream_frames(self, path_or_id: str, batch_size: int = 32) -> AsyncIterator[bytes]:
        """Stream decoded video frames optimized for ML processing."""
        pass

    @abc.abstractmethod
    async def get_metadata(self, path_or_id: str) -> dict:
        """Get video metadata including frame count, FPS, etc."""
        pass
