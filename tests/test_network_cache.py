import os
import time
import shutil
import tempfile
import threading
from pathlib import Path
from unittest import mock
import pytest
import redis
import zmq
import numpy as np
from raad.clients.network_cache_client import NetworkCacheClient, CacheStats
from raad.clients.remote_client import RemoteVideoClient
from raad.catalog import VideoCatalog, VideoMetadata

class MockRedis:
    def __init__(self):
        self.data = {}
        self.sets = {}
        self.sorted_sets = {}
    
    def incr(self, key):
        self.data[key] = self.data.get(key, 0) + 1
        return self.data[key]
    
    def zadd(self, name, mapping):
        if name not in self.sorted_sets:
            self.sorted_sets[name] = {}
        self.sorted_sets[name].update(mapping)
    
    def zincrby(self, name, amount, value):
        if name not in self.sorted_sets:
            self.sorted_sets[name] = {}
        self.sorted_sets[name][value] = self.sorted_sets[name].get(value, 0) + amount
        return self.sorted_sets[name][value]
    
    def smembers(self, key):
        return self.sets.get(key, set())
    
    def sadd(self, key, *values):
        if key not in self.sets:
            self.sets[key] = set()
        self.sets[key].update(values)

@pytest.fixture
def mock_redis():
    return MockRedis()

@pytest.fixture
def temp_cache_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_remote_client():
    client = mock.Mock(spec=RemoteVideoClient)
    client.exists.return_value = True
    client.download.side_effect = lambda video_id, dest: Path(dest).write_bytes(b"mock video data")
    return client

@pytest.fixture
def network_cache_client(temp_cache_dir, mock_remote_client, mock_redis):
    with mock.patch('redis.from_url', return_value=mock_redis):
        with mock.patch('zmq.Context'):
            client = NetworkCacheClient(
                remote_client=mock_remote_client,
                cache_dir=temp_cache_dir,
                redis_url="redis://fake:6379",
                max_cache_size_gb=1
            )
            yield client

class TestNetworkCacheClient:
    def test_initialization(self, network_cache_client, temp_cache_dir):
        """Test client initialization."""
        assert network_cache_client.cache_dir == Path(temp_cache_dir)
        assert network_cache_client.stats.hits == 0
        assert network_cache_client.stats.misses == 0

    def test_cache_path_generation(self, network_cache_client):
        """Test content-based hash path generation."""
        video_id = "test_video_123"
        cache_path = network_cache_client._get_cache_path(video_id)
        assert isinstance(cache_path, Path)
        # Verify two-level directory structure
        assert len(cache_path.parts) >= 3

    def test_download_uncached_video(self, network_cache_client, mock_remote_client):
        """Test downloading an uncached video."""
        video_id = "uncached_video"
        dest_path = tempfile.mktemp()
        
        network_cache_client.download(video_id, dest_path)
        
        # Verify remote client was called
        mock_remote_client.download.assert_called_once()
        assert Path(dest_path).exists()
        assert network_cache_client.stats.misses == 1
        assert network_cache_client.stats.hits == 0

    def test_download_cached_video(self, network_cache_client, mock_remote_client):
        """Test downloading a cached video."""
        video_id = "cached_video"
        dest_path = tempfile.mktemp()
        
        # First download to cache
        network_cache_client.download(video_id, dest_path)
        mock_remote_client.download.reset_mock()
        
        # Second download should use cache
        second_dest = tempfile.mktemp()
        network_cache_client.download(video_id, second_dest)
        
        # Verify remote client wasn't called again
        mock_remote_client.download.assert_not_called()
        assert Path(second_dest).exists()
        assert network_cache_client.stats.hits == 1
        assert network_cache_client.stats.misses == 1

    @pytest.mark.asyncio
    async def test_concurrent_downloads(self, network_cache_client):
        """Test concurrent downloads of the same video."""
        video_id = "concurrent_test"
        num_concurrent = 5
        dest_paths = [tempfile.mktemp() for _ in range(num_concurrent)]
        
        async def download(dest):
            network_cache_client.download(video_id, dest)
        
        import asyncio
        await asyncio.gather(*[download(dest) for dest in dest_paths])
        
        # Verify all downloads completed
        assert all(Path(dest).exists() for dest in dest_paths)
        # Should only have one cache miss despite concurrent downloads
        assert network_cache_client.stats.misses == 1

    def test_cache_eviction(self, network_cache_client, mock_remote_client):
        """Test cache size maintenance and eviction."""
        # Create large mock videos
        large_size = 1024 * 1024 * 100  # 100MB
        mock_remote_client.download.side_effect = lambda _, dest: Path(dest).write_bytes(b"x" * large_size)
        
        # Download multiple large videos
        for i in range(15):
            network_cache_client.download(f"large_video_{i}", tempfile.mktemp())
            time.sleep(0.1)  # Ensure different access times
        
        # Verify cache size is maintained
        cache_size = sum(f.stat().st_size for f in Path(network_cache_client.cache_dir).rglob("*") if f.is_file())
        assert cache_size <= network_cache_client.max_cache_size

    def test_peer_discovery(self, network_cache_client):
        """Test peer discovery and updates."""
        with mock.patch.object(network_cache_client, 'discovery_socket') as mock_socket:
            network_cache_client._handle_peer_discovery()
            mock_socket.send_json.assert_called_once()
            sent_data = mock_socket.send_json.call_args[0][0]
            assert 'peer_id' in sent_data
            assert 'cached_videos' in sent_data

    def test_stats_reporting(self, network_cache_client):
        """Test statistics reporting."""
        # Generate some activity
        network_cache_client.download("stats_test_1", tempfile.mktemp())
        network_cache_client.download("stats_test_1", tempfile.mktemp())
        network_cache_client.download("stats_test_2", tempfile.mktemp())
        
        stats = network_cache_client.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 2
        assert stats['hit_ratio'] == pytest.approx(0.333, rel=0.01)
        assert 'cache_size' in stats
        assert 'prefetch_queue_size' in stats

    @pytest.mark.parametrize("error_type", [
        redis.ConnectionError,
        zmq.ZMQError,
        OSError
    ])
    def test_error_handling(self, network_cache_client, error_type):
        """Test handling of various error conditions."""
        with mock.patch.object(network_cache_client, '_fetch_from_best_source', side_effect=error_type):
            with pytest.raises(Exception) as exc_info:
                network_cache_client.download("error_test", tempfile.mktemp())
            assert "Download failed" in str(exc_info.value)

class TestNetworkCachePerformance:
    @pytest.mark.benchmark
    def test_download_performance(self, network_cache_client, benchmark):
        """Benchmark download performance."""
        def download_video():
            network_cache_client.download("bench_test", tempfile.mktemp())
        
        result = benchmark(download_video)
        assert result.stats.mean < 1.0  # Should complete within 1 second

    @pytest.mark.benchmark
    def test_concurrent_performance(self, network_cache_client, benchmark):
        """Benchmark concurrent download performance."""
        num_concurrent = 10
        
        def concurrent_downloads():
            threads = []
            for i in range(num_concurrent):
                thread = threading.Thread(
                    target=network_cache_client.download,
                    args=(f"concurrent_bench_{i}", tempfile.mktemp())
                )
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
        
        result = benchmark(concurrent_downloads)
        assert result.stats.mean < 5.0  # Should complete within 5 seconds

if __name__ == '__main__':
    pytest.main([__file__])
