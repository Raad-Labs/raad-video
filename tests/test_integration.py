import os
import tempfile
import threading
import time
from pathlib import Path
import pytest
import numpy as np
import cv2
from raad.data_loader import (
    VideoDataLoader, ProcessingMode, FrameFormat, ColorSpace,
    StreamingConfig, CacheConfig, DistributedConfig
)
from raad.catalog import VideoCatalog, VideoMetadata, DatasetSplit
from raad.clients.network_cache_client import NetworkCacheClient
from raad.clients.remote_client import RemoteVideoClient

@pytest.fixture
def sample_video():
    """Create a sample video file for testing."""
    temp_dir = tempfile.mkdtemp()
    video_path = Path(temp_dir) / "test_video.mp4"
    
    # Create a simple video with colored frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    
    # Generate 90 frames (3 seconds)
    for i in range(90):
        # Create colored frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        color = int(255 * (i / 90))
        frame[:, :, 0] = color  # Vary blue channel
        out.write(frame)
    
    out.release()
    
    yield video_path
    # Cleanup
    if video_path.exists():
        video_path.unlink()
    os.rmdir(temp_dir)

@pytest.fixture
def test_catalog(sample_video):
    """Create a test video catalog."""
    catalog = VideoCatalog()
    metadata = VideoMetadata(
        video_id="test_video_1",
        local_path=str(sample_video),
        remote_path="videos/test_video_1.mp4",
        dataset_split=DatasetSplit.TRAIN,
        duration=3.0,
        fps=30.0,
        resolution=(640, 480),
        categories=["test"]
    )
    catalog.add_video(metadata)
    return catalog

@pytest.fixture
def network_cache_setup():
    """Setup network cache infrastructure."""
    cache_dir = tempfile.mkdtemp()
    remote_client = RemoteVideoClient("http://localhost:8000")
    
    network_client = NetworkCacheClient(
        remote_client=remote_client,
        cache_dir=cache_dir,
        redis_url="redis://localhost:6379",
        max_cache_size_gb=1
    )
    
    yield network_client
    
    # Cleanup
    import shutil
    shutil.rmtree(cache_dir)

class TestVideoProcessingPipeline:
    def test_end_to_end_processing(self, test_catalog, network_cache_setup, sample_video):
        """Test the entire video processing pipeline."""
        loader = VideoDataLoader(
            catalog=test_catalog,
            client=network_cache_setup,
            processing_mode=ProcessingMode.MULTI_THREAD,
            frame_format=FrameFormat.NUMPY,
            color_space=ColorSpace.RGB,
            streaming_config=StreamingConfig(
                mode=StreamingMode.ADAPTIVE,
                buffer_size=1000
            ),
            cache_config=CacheConfig(
                enabled=True,
                max_size=1000,
                policy='lru'
            ),
            batch_size=16,
            num_workers=2,
            device="cpu"
        )
        
        # Process video and collect frames
        frames = []
        for batch in loader.get_dataset_iterator(DatasetSplit.TRAIN):
            if isinstance(batch, list):
                frames.extend(batch)
            else:
                frames.append(batch)
        
        # Verify frame count
        assert len(frames) == 90  # 3 seconds * 30 fps
        
        # Verify frame properties
        assert frames[0].shape == (480, 640, 3)
        assert frames[-1].shape == (480, 640, 3)
        
        # Verify color progression
        first_frame_blue = frames[0][:, :, 0].mean()
        last_frame_blue = frames[-1][:, :, 0].mean()
        assert last_frame_blue > first_frame_blue

    def test_distributed_processing(self, test_catalog, network_cache_setup):
        """Test distributed video processing."""
        # Setup distributed configuration
        distributed_config = DistributedConfig(
            enabled=True,
            num_nodes=2,
            node_rank=0
        )
        
        loader = VideoDataLoader(
            catalog=test_catalog,
            client=network_cache_setup,
            processing_mode=ProcessingMode.DISTRIBUTED,
            distributed_config=distributed_config,
            batch_size=16
        )
        
        # Simulate distributed processing
        processed_frames = []
        for batch in loader.get_dataset_iterator(DatasetSplit.TRAIN):
            processed_frames.extend(batch if isinstance(batch, list) else [batch])
        
        assert len(processed_frames) > 0

    def test_streaming_performance(self, test_catalog, network_cache_setup):
        """Test streaming performance under load."""
        loader = VideoDataLoader(
            catalog=test_catalog,
            client=network_cache_setup,
            streaming_config=StreamingConfig(
                mode=StreamingMode.REAL_TIME,
                max_latency=0.1
            ),
            batch_size=1  # Process frame-by-frame for latency testing
        )
        
        # Measure frame processing latencies
        latencies = []
        start_time = time.time()
        
        for frame in loader.get_dataset_iterator(DatasetSplit.TRAIN):
            latency = time.time() - start_time
            latencies.append(latency)
            start_time = time.time()
        
        # Verify latency requirements
        assert max(latencies) < 0.2  # Maximum latency threshold
        assert sum(latencies) / len(latencies) < 0.1  # Average latency threshold

    def test_error_recovery(self, test_catalog, network_cache_setup):
        """Test error recovery and fault tolerance."""
        loader = VideoDataLoader(
            catalog=test_catalog,
            client=network_cache_setup,
            processing_mode=ProcessingMode.MULTI_THREAD
        )
        
        # Simulate network failures
        def interrupt_processing():
            time.sleep(0.5)
            network_cache_setup.remote_client.download.side_effect = ConnectionError
            time.sleep(0.5)
            network_cache_setup.remote_client.download.side_effect = None
        
        threading.Thread(target=interrupt_processing, daemon=True).start()
        
        # Process video despite interruptions
        frames = []
        for batch in loader.get_dataset_iterator(DatasetSplit.TRAIN):
            if isinstance(batch, list):
                frames.extend(batch)
            else:
                frames.append(batch)
        
        # Verify we got all frames despite errors
        assert len(frames) == 90

    @pytest.mark.benchmark
    def test_processing_performance(self, test_catalog, network_cache_setup, benchmark):
        """Benchmark video processing performance."""
        loader = VideoDataLoader(
            catalog=test_catalog,
            client=network_cache_setup,
            processing_mode=ProcessingMode.MULTI_THREAD,
            batch_size=32,
            num_workers=4
        )
        
        def process_video():
            frames = []
            for batch in loader.get_dataset_iterator(DatasetSplit.TRAIN):
                if isinstance(batch, list):
                    frames.extend(batch)
                else:
                    frames.append(batch)
            return len(frames)
        
        result = benchmark(process_video)
        assert result.stats.mean < 2.0  # Should process video within 2 seconds

if __name__ == '__main__':
    pytest.main([__file__])
