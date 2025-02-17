# clients/remote_client.py
import os
from pathlib import Path
from typing import Optional, AsyncIterator, Dict, Any
import asyncio
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import StreamDataReceived
from aioquic.h3.connection import H3_ALPN
import aiohttp
import numpy as np
import av
from .base_client import BaseVideoClient, TransportProtocol

class RemoteClientError(Exception):
    """Base exception for remote client errors."""
    pass

class RemoteVideoClient(BaseVideoClient):
    """
    Implements video streaming with support for QUIC and WebTransport protocols.
    Optimized for ML workloads with efficient frame batching and preprocessing.
    """
    def __init__(
        self,
        base_url: str,
        auth_token: Optional[str] = None,
        transport: TransportProtocol = TransportProtocol.QUIC,
        timeout: int = 30,
        batch_size: int = 32,
        prefetch_size: int = 64
    ):
        super().__init__(transport)
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.timeout = timeout
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        
        # Initialize transport-specific configurations
        if transport == TransportProtocol.QUIC:
            self.quic_config = QuicConfiguration(
                alpn_protocols=H3_ALPN,
                is_client=True
            )
        
        # Initialize HTTP client session
        self.session = None

    async def _init_session(self):
        """Initialize the appropriate client session based on transport."""
        if not self.session:
            if self.transport == TransportProtocol.HTTP:
                self.session = aiohttp.ClientSession()
            elif self.transport == TransportProtocol.QUIC:
                self.session = await self._init_quic_session()
            elif self.transport == TransportProtocol.WEBTRANSPORT:
                self.session = await self._init_webtransport_session()

    async def _init_quic_session(self):
        """Initialize QUIC session with the server."""
        loop = asyncio.get_event_loop()
        return await loop.create_connection(
            lambda: QuicConnectionProtocol(self.quic_config),
            self.base_url.split('://')[1],
            443
        )

    async def _init_webtransport_session(self):
        """Initialize WebTransport session."""
        # WebTransport initialization will go here
        pass

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with optional authentication."""
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    async def exists(self, path_or_id: str) -> bool:
        """Check if remote resource exists."""
        await self._init_session()
        try:
            async with self.session.head(f"{self.base_url}/{path_or_id}") as response:
                return response.status == 200
        except Exception:
            return False

    async def download(self, path_or_id: str, destination: str) -> None:
        """Download with support for modern transport protocols."""
        await self._init_session()
        Path(destination).parent.mkdir(parents=True, exist_ok=True)

        async with self.session.get(
            f"{self.base_url}/{path_or_id}",
            headers=self._get_headers()
        ) as response:
            with open(destination, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)

    async def stream(self, path_or_id: str, batch_size: int = None) -> AsyncIterator[bytes]:
        """Stream video data with protocol-specific optimizations."""
        batch_size = batch_size or self.batch_size
        await self._init_session()

        async with self.session.get(
            f"{self.base_url}/{path_or_id}/stream",
            headers=self._get_headers()
        ) as response:
            async for chunk in response.content.iter_chunked(batch_size):
                yield chunk

    async def stream_frames(
        self,
        path_or_id: str,
        batch_size: int = None,
        use_gpu: bool = True,
        compression: str = "lz4",
        quality: int = 80
    ) -> AsyncIterator[np.ndarray]:
        """Stream frames with maximum performance using GPU acceleration."""
        batch_size = batch_size or self.batch_size
        
        # Initialize GPU resources if available
        if use_gpu and torch.cuda.is_available():
            torch_stream = torch.cuda.Stream()
            gpu_buffer = torch.cuda.Stream()
        else:
            use_gpu = False
        
        # Initialize frame buffer for zero-copy transfers
        frame_buffer = ZeroCopyFrameBuffer()
        
        async with self.session.get(
            f"{self.base_url}/{path_or_id}/frames",
            params={
                'batch_size': batch_size,
                'use_gpu': use_gpu,
                'compression': compression,
                'quality': quality
            },
            headers=self._get_headers()
        ) as response:
            while True:
                # Read metadata
                metadata_line = await response.content.readline()
                if not metadata_line:
                    break
                    
                metadata = json.loads(metadata_line)
                frame_size = metadata['size']
                frame_shape = metadata['shape']
                
                # Read frame data using zero-copy when possible
                frame_data = await response.content.read(frame_size)
                
                # Process based on compression method
                if metadata['compression'] == 'lz4':
                    decompressed = lz4.frame.decompress(frame_data)
                    if use_gpu:
                        with torch.cuda.stream(torch_stream):
                            # Move directly to GPU
                            frames = torch.frombuffer(
                                decompressed,
                                dtype=torch.uint8
                            ).reshape(frame_shape).cuda(non_blocking=True)
                    else:
                        frames = np.frombuffer(
                            decompressed,
                            dtype=np.uint8
                        ).reshape(frame_shape)
                        
                elif metadata['compression'] == 'nvjpeg' and use_gpu:
                    with torch.cuda.stream(gpu_buffer):
                        frames = nvjpeg_handle.decode(
                            frame_data,
                            device='cuda'
                        )
                else:
                    if use_gpu:
                        with torch.cuda.stream(torch_stream):
                            frames = torch.frombuffer(
                                frame_data,
                                dtype=torch.uint8
                            ).reshape(frame_shape).cuda(non_blocking=True)
                    else:
                        frames = np.frombuffer(
                            frame_data,
                            dtype=np.uint8
                        ).reshape(frame_shape)
                
                # Optional: Apply any additional preprocessing
                if use_gpu:
                    with torch.cuda.stream(torch_stream):
                        frames = frames.float() / 255.0  # Normalize on GPU
                        # Add any other preprocessing steps here
                else:
                    frames = frames.astype(np.float32) / 255.0
                
                yield frames

    async def get_metadata(self, path_or_id: str) -> dict:
        """Get video metadata."""
        await self._init_session()
        async with self.session.get(
            f"{self.base_url}/{path_or_id}/info",
            headers=self._get_headers()
        ) as response:
            return await response.json()

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
