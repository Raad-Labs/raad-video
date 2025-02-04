import os
import time
import json
import hashlib
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple, Generator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import zmq
import redis
from .base_client import BaseVideoClient
from .remote_client import RemoteVideoClient
from .local_client import LocalVideoClient

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    network_bytes_saved: int = 0
    last_prefetch_time: float = 0
    prefetch_queue_size: int = 0

class NetworkCacheClient(BaseVideoClient):
    def stream(self, video_id: str) -> Generator[bytes, None, None]:
        """Stream video data from cache or remote source.
        
        Args:
            video_id: Identifier of the video to stream
            
        Yields:
            Chunks of video data as bytes
        """
        cache_path = self._get_cache_path(video_id)
        
        if self._is_cached_locally(video_id):
            # Stream from local cache
            with open(cache_path, 'rb') as f:
                while chunk := f.read(8192):  # 8KB chunks
                    yield chunk
            self.stats.hits += 1
        else:
            # Stream from remote and cache simultaneously
            self.stats.misses += 1
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, 'wb') as cache_file:
                for chunk in self.remote_client.stream(video_id):
                    cache_file.write(chunk)
                    yield chunk
    """
    Sophisticated network caching client that provides:
    1. Local network caching with Redis
    2. Smart prefetching based on access patterns
    3. ZeroMQ-based peer-to-peer sharing
    4. Bandwidth-aware data transfer
    5. Cache coherency management
    """
    
    def __init__(
        self,
        remote_client: RemoteVideoClient,
        cache_dir: str,
        redis_url: str = "redis://localhost:6379",
        zmq_port: int = 5555,
        max_cache_size_gb: float = 100,
        prefetch_thread_count: int = 4,
        peer_discovery_port: int = 5556
    ):
        self.remote_client = remote_client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Redis connection for metadata and access patterns
        self.redis = redis.from_url(redis_url)
        
        # Initialize ZMQ for peer-to-peer sharing
        self.context = zmq.Context()
        self.setup_zmq_sockets(zmq_port, peer_discovery_port)
        
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.stats = CacheStats()
        
        # Start background workers
        self.prefetch_queue: List[str] = []
        self.prefetch_lock = threading.Lock()
        self.prefetch_executor = ThreadPoolExecutor(max_workers=prefetch_thread_count)
        self.start_background_tasks()

    def setup_zmq_sockets(self, zmq_port: int, discovery_port: int) -> None:
        """Setup ZMQ sockets for P2P communication."""
        # Data transfer socket
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{zmq_port}")
        
        # Peer discovery socket
        self.discovery_socket = self.context.socket(zmq.PUB)
        self.discovery_socket.bind(f"tcp://*:{discovery_port}")
        
        # Start socket monitoring threads
        threading.Thread(target=self._handle_data_requests, daemon=True).start()
        threading.Thread(target=self._handle_peer_discovery, daemon=True).start()

    def start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        threading.Thread(target=self._maintain_cache_size, daemon=True).start()
        threading.Thread(target=self._update_access_patterns, daemon=True).start()
        threading.Thread(target=self._prefetch_worker, daemon=True).start()

    def _get_cache_path(self, video_id: str) -> Path:
        """Get cache path for a video using content-based hashing."""
        hash_obj = hashlib.sha256(video_id.encode())
        hash_str = hash_obj.hexdigest()
        # Create two-level directory structure to avoid too many files in one directory
        return self.cache_dir / hash_str[:2] / hash_str[2:4] / hash_str

    def _is_cached_locally(self, video_id: str) -> bool:
        """Check if video is cached locally."""
        cache_path = self._get_cache_path(video_id)
        return cache_path.exists()

    def _get_peer_locations(self, video_id: str) -> List[str]:
        """Get list of peers that have the video cached."""
        return self.redis.smembers(f"peers:{video_id}")

    def _update_access_patterns(self) -> None:
        """Update video access patterns for smart prefetching."""
        while True:
            # Update access frequency and recency
            for video_id in self.redis.scan_iter("access:*"):
                freq = self.redis.zincrby("video_frequency", 1, video_id)
                self.redis.zadd("video_recency", {video_id: time.time()})
                
                # Update prefetch queue based on frequency and recency
                if freq > 5:  # Threshold for prefetching
                    with self.prefetch_lock:
                        if video_id not in self.prefetch_queue:
                            self.prefetch_queue.append(video_id)
            
            time.sleep(60)  # Update every minute

    def _maintain_cache_size(self) -> None:
        """Maintain cache size within limits using LRU policy."""
        while True:
            total_size = sum(f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file())
            
            if total_size > self.max_cache_size:
                # Get least recently used files
                lru_files = sorted(
                    self.cache_dir.rglob("*"),
                    key=lambda x: x.stat().st_atime if x.is_file() else float('inf')
                )
                
                for file in lru_files:
                    if file.is_file():
                        file_size = file.stat().st_size
                        file.unlink()
                        total_size -= file_size
                        
                        if total_size <= self.max_cache_size * 0.9:  # Add 10% buffer
                            break
            
            time.sleep(300)  # Check every 5 minutes

    def _prefetch_worker(self) -> None:
        """Worker for prefetching videos."""
        while True:
            if self.prefetch_queue:
                with self.prefetch_lock:
                    video_id = self.prefetch_queue.pop(0)
                    self.stats.prefetch_queue_size = len(self.prefetch_queue)
                
                if not self._is_cached_locally(video_id):
                    try:
                        self._fetch_from_best_source(video_id)
                    except Exception as e:
                        print(f"Prefetch error for {video_id}: {e}")
            
            time.sleep(1)

    def _handle_data_requests(self) -> None:
        """Handle incoming P2P data requests."""
        while True:
            try:
                message = self.socket.recv_json()
                video_id = message.get("video_id")
                
                if video_id and self._is_cached_locally(video_id):
                    cache_path = self._get_cache_path(video_id)
                    with open(cache_path, "rb") as f:
                        data = f.read()
                    self.socket.send(data)
                else:
                    self.socket.send_json({"error": "Video not found"})
            except Exception as e:
                print(f"Error handling data request: {e}")

    def _handle_peer_discovery(self) -> None:
        """Handle peer discovery and updates."""
        while True:
            # Broadcast cached videos to peers
            cached_videos = [str(p.name) for p in self.cache_dir.rglob("*") if p.is_file()]
            self.discovery_socket.send_json({
                "peer_id": id(self),
                "cached_videos": cached_videos
            })
            time.sleep(60)

    def _fetch_from_best_source(self, video_id: str) -> None:
        """Fetch video from the best available source (peer or remote)."""
        peers = self._get_peer_locations(video_id)
        
        if peers:
            # Try to fetch from nearest peer first
            for peer in peers:
                try:
                    peer_socket = self.context.socket(zmq.REQ)
                    peer_socket.connect(f"tcp://{peer}")
                    peer_socket.send_json({"video_id": video_id})
                    
                    data = peer_socket.recv()
                    if not isinstance(data, dict):  # Not an error response
                        cache_path = self._get_cache_path(video_id)
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(cache_path, "wb") as f:
                            f.write(data)
                        return
                except Exception as e:
                    print(f"Peer fetch error: {e}")
                finally:
                    peer_socket.close()
        
        # Fallback to remote fetch if peer fetch fails
        destination = self._get_cache_path(video_id)
        self.remote_client.download(video_id, str(destination))

    def exists(self, video_id: str) -> bool:
        """Check if video exists in cache or remote."""
        return self._is_cached_locally(video_id) or self.remote_client.exists(video_id)

    def download(self, video_id: str, destination: str) -> None:
        """Download video with caching."""
        try:
            cache_path = self._get_cache_path(video_id)
            
            # Update access patterns
            self.redis.incr(f"access:{video_id}")
            
            if self._is_cached_locally(video_id):
                # Serve from local cache
                with open(cache_path, "rb") as src, open(destination, "wb") as dst:
                    dst.write(src.read())
                self.stats.hits += 1
                self.stats.network_bytes_saved += cache_path.stat().st_size
            else:
                # Fetch from best source
                self.stats.misses += 1
                self._fetch_from_best_source(video_id)
                with open(cache_path, "rb") as src, open(destination, "wb") as dst:
                    dst.write(src.read())
        
        except Exception as e:
            raise Exception(f"Download failed for {video_id}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_ratio": self.stats.hits / (self.stats.hits + self.stats.misses) if (self.stats.hits + self.stats.misses) > 0 else 0,
            "network_bytes_saved": self.stats.network_bytes_saved,
            "cache_size": sum(f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file()),
            "prefetch_queue_size": self.stats.prefetch_queue_size,
            "peer_count": len(self.redis.smembers("peers")),
        }
