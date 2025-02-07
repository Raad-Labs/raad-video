# data_loader.py
import os
import shutil
import tempfile
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, Manager
import threading
from queue import Queue
from collections import deque
import cv2
import numpy as np
from typing import Iterator, List, Optional, Tuple, Dict, Any, Union, Callable, Set
from abc import ABC, abstractmethod
from .catalog import VideoCatalog, VideoMetadata, DatasetSplit
from .clients.base_client import BaseVideoClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoLoadError(Exception):
    """Raised when there's an error loading or processing a video.
    
    This exception is raised in cases such as:
    - File not found or inaccessible
    - Corrupted video file
    - Insufficient memory for processing
    - GPU errors during processing
    - Network errors in distributed mode
    """
    pass

class ProcessingMode(Enum):
    """Video processing modes for different performance requirements.
    
    Modes:
        SINGLE_PROCESS: Sequential processing in a single process. Good for debugging.
        MULTI_THREAD: Parallel processing using threads. Best for I/O-bound tasks.
        MULTI_PROCESS: Parallel processing using multiple processes. Best for CPU-bound tasks.
        DISTRIBUTED: Distributed processing across multiple nodes. Best for large-scale processing.
    """
    SINGLE_PROCESS = auto()
    MULTI_THREAD = auto()
    MULTI_PROCESS = auto()
    DISTRIBUTED = auto()

class FrameFormat(Enum):
    """Supported frame formats for different ML frameworks.
    
    Formats:
        NUMPY: Standard NumPy array format (H, W, C)
        TORCH: PyTorch tensor format (C, H, W)
        TENSORFLOW: TensorFlow tensor format (H, W, C)
        JAX: JAX DeviceArray format
        PADDLE: PaddlePaddle tensor format
        ONNX: ONNX tensor format
        PIL: PIL Image format
    
    Note: All formats maintain consistent channel ordering (RGB/BGR) as specified
    in the color_space parameter.
    """
    NUMPY = auto()
    TORCH = auto()
    TENSORFLOW = auto()
    JAX = auto()
    PADDLE = auto()
    ONNX = auto()
    PIL = auto()

class ColorSpace(Enum):
    """Supported color spaces."""
    RGB = auto()
    BGR = auto()
    GRAYSCALE = auto()
    HSV = auto()
    LAB = auto()
    YUV = auto()

class StreamingMode(Enum):
    """Video streaming modes."""
    SEQUENTIAL = auto()  # Standard sequential processing
    ADAPTIVE = auto()    # Adaptive streaming based on system load
    PRIORITY = auto()    # Priority-based streaming
    REAL_TIME = auto()   # Real-time streaming with frame dropping

@dataclass
class ProcessingMetrics:
    """Metrics for monitoring video processing performance.
    
    Attributes:
        total_frames: Total number of frames in the video(s)
        processed_frames: Number of successfully processed frames
        dropped_frames: Number of frames dropped due to performance constraints
        avg_processing_time: Average time (seconds) to process a single frame
        peak_memory_usage: Peak memory usage in bytes during processing
        cache_hits: Number of successful cache retrievals
        cache_misses: Number of failed cache retrievals
        start_time: Processing start time (Unix timestamp)
    
    Note: These metrics are continuously updated during processing and can be
    exported for monitoring and optimization purposes.
    """
    total_frames: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0
    avg_processing_time: float = 0.0
    peak_memory_usage: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    start_time: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fps": self.processed_frames / (time.time() - self.start_time),
            "progress": self.processed_frames / max(1, self.total_frames),
            "drop_rate": self.dropped_frames / max(1, self.total_frames),
            "avg_processing_time": self.avg_processing_time,
            "peak_memory_mb": self.peak_memory_usage / (1024 * 1024),
            "cache_hit_ratio": self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }

class AugmentationPipeline(ABC):
    """Abstract base class for augmentation pipelines."""
    @abstractmethod
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

@dataclass
class StandardAugmentationPipeline(AugmentationPipeline):
    """Standard augmentation pipeline with advanced configurations."""
    # Geometric transformations
    horizontal_flip: bool = False
    vertical_flip: bool = False
    rotation_range: Tuple[float, float] = (0, 0)
    shear_range: Tuple[float, float] = (0, 0)
    zoom_range: Tuple[float, float] = (1.0, 1.0)
    
    # Color transformations
    brightness_range: Tuple[float, float] = (1.0, 1.0)
    contrast_range: Tuple[float, float] = (1.0, 1.0)
    saturation_range: Tuple[float, float] = (1.0, 1.0)
    hue_range: Tuple[float, float] = (0, 0)
    
    # Noise and artifacts
    noise_types: Set[str] = field(default_factory=lambda: set())
    noise_params: Dict[str, Any] = field(default_factory=dict)
    blur_types: Set[str] = field(default_factory=lambda: set())
    blur_params: Dict[str, Any] = field(default_factory=dict)
    
    # Advanced transformations
    elastic_transform: bool = False
    elastic_params: Dict[str, Any] = field(default_factory=dict)
    perspective_transform: bool = False
    perspective_params: Dict[str, Any] = field(default_factory=dict)
    
    # Cropping and resizing
    random_crop: Optional[Tuple[int, int]] = None
    random_erasing: bool = False
    erasing_params: Dict[str, Any] = field(default_factory=dict)
    
    # Mixing strategies
    mixup: bool = False
    cutmix: bool = False
    mosaic: bool = False
    mixing_params: Dict[str, Any] = field(default_factory=dict)
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Apply the augmentation pipeline to a frame."""
        # Implementation in the class below
        return frame

    def get_config(self) -> Dict[str, Any]:
        return {
            "geometric": {
                "horizontal_flip": self.horizontal_flip,
                "vertical_flip": self.vertical_flip,
                "rotation_range": self.rotation_range,
                "shear_range": self.shear_range,
                "zoom_range": self.zoom_range
            },
            "color": {
                "brightness_range": self.brightness_range,
                "contrast_range": self.contrast_range,
                "saturation_range": self.saturation_range,
                "hue_range": self.hue_range
            },
            "noise": {
                "types": list(self.noise_types),
                "params": self.noise_params
            },
            "blur": {
                "types": list(self.blur_types),
                "params": self.blur_params
            },
            "advanced": {
                "elastic": {
                    "enabled": self.elastic_transform,
                    "params": self.elastic_params
                },
                "perspective": {
                    "enabled": self.perspective_transform,
                    "params": self.perspective_params
                }
            },
            "cropping": {
                "random_crop": self.random_crop,
                "random_erasing": {
                    "enabled": self.random_erasing,
                    "params": self.erasing_params
                }
            },
            "mixing": {
                "mixup": self.mixup,
                "cutmix": self.cutmix,
                "mosaic": self.mosaic,
                "params": self.mixing_params
            }
        }

@dataclass
class DistributedConfig:
    """Configuration for distributed processing."""
    enabled: bool = False
    num_nodes: int = 1
    node_rank: int = 0
    master_addr: str = 'localhost'
    master_port: int = 29500
    backend: str = 'gloo'
    init_method: Optional[str] = None
    world_size: Optional[int] = None

@dataclass
class CacheConfig:
    """Configuration for frame caching."""
    enabled: bool = True
    max_size: int = 1000
    policy: str = 'lru'  # 'lru', 'lfu', 'fifo'
    persistent: bool = False
    cache_dir: Optional[str] = None
    compression: Optional[str] = None  # 'lz4', 'zstd', None

@dataclass
class StreamingConfig:
    """Configuration for video streaming."""
    mode: StreamingMode = StreamingMode.SEQUENTIAL
    buffer_size: int = 1000
    max_latency: float = 0.1  # seconds
    drop_threshold: float = 0.8  # drop frames if buffer is above this
    priority_key: Optional[Callable] = None  # for priority-based streaming

class VideoDataLoader:
    """
    State-of-the-art video data loader with support for:
    1. Distributed processing across multiple nodes
    2. Multiple ML framework outputs (PyTorch, TF, JAX, etc.)
    3. Advanced augmentation pipeline with mixing strategies
    4. Smart caching with multiple policies
    5. Adaptive streaming modes
    6. Multi-GPU support with memory optimization
    7. Real-time performance monitoring
    8. Auto-tuning capabilities
    """
    SUPPORTED_FORMATS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.3gp')

    def __init__(
        self, 
        catalog: VideoCatalog, 
        client: BaseVideoClient,
        processing_mode: ProcessingMode = ProcessingMode.MULTI_THREAD,
        frame_format: FrameFormat = FrameFormat.NUMPY,
        color_space: ColorSpace = ColorSpace.RGB,
        streaming_config: Optional[StreamingConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        distributed_config: Optional[DistributedConfig] = None,
        frame_skip: int = 1,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        batch_size: Optional[int] = None,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        device: Optional[str] = None,
        devices: Optional[List[str]] = None,  # For multi-GPU
        augmentation: Optional[AugmentationPipeline] = None,
        seed: Optional[int] = None,
        auto_tune: bool = False,
        monitoring_interval: float = 1.0,  # seconds
        export_metrics: bool = False,
        metrics_path: Optional[str] = None
    ) -> None:
        # Basic configuration
        self.catalog = catalog
        self.client = client
        self.processing_mode = processing_mode
        self.frame_format = frame_format
        self.color_space = color_space
        self.frame_skip = frame_skip
        self.target_size = target_size
        self.normalize = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        # Advanced configuration
        self.streaming_config = streaming_config or StreamingConfig()
        self.cache_config = cache_config or CacheConfig()
        self.distributed_config = distributed_config or DistributedConfig()
        self.auto_tune = auto_tune
        self.monitoring_interval = monitoring_interval
        self.export_metrics = export_metrics
        self.metrics_path = metrics_path

        # GPU configuration
        self.device = device
        self.devices = devices or []
        if self.devices and not self.device:
            self.device = self.devices[0]

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            if self.frame_format == FrameFormat.TORCH:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

        # Initialize processing components
        self._setup_processing_environment()
        self._setup_caching()
        self._setup_monitoring()
        
        # Initialize metrics
        self.metrics = ProcessingMetrics()
        
        # Start monitoring if enabled
        if self.export_metrics:
            self._start_metrics_export()

    def _setup_processing_environment(self) -> None:
        """Setup processing environment based on mode."""
        if self.processing_mode == ProcessingMode.DISTRIBUTED:
            self._setup_distributed()
        elif self.processing_mode == ProcessingMode.MULTI_PROCESS:
            self.process_pool = Pool(self.num_workers)
        elif self.processing_mode == ProcessingMode.MULTI_THREAD:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.num_workers)

        # Setup streaming components
        self.frame_queue = Queue(maxsize=self.streaming_config.buffer_size)
        self.priority_queue = [] if self.streaming_config.mode == StreamingMode.PRIORITY else None

    def _setup_distributed(self) -> None:
        """Setup distributed processing environment."""
        if not self.distributed_config.enabled:
            return

        if self.frame_format == FrameFormat.TORCH:
            import torch.distributed as dist
            if not dist.is_initialized():
                if self.distributed_config.init_method:
                    dist.init_process_group(
                        backend=self.distributed_config.backend,
                        init_method=self.distributed_config.init_method,
                        world_size=self.distributed_config.world_size or self.distributed_config.num_nodes,
                        rank=self.distributed_config.node_rank
                    )
                else:
                    dist.init_process_group(
                        backend=self.distributed_config.backend,
                        init_method=f'tcp://{self.distributed_config.master_addr}:{self.distributed_config.master_port}',
                        world_size=self.distributed_config.num_nodes,
                        rank=self.distributed_config.node_rank
                    )

    def _setup_caching(self) -> None:
        """Setup frame caching system."""
        if not self.cache_config.enabled:
            return

        if self.cache_config.persistent and self.cache_config.cache_dir:
            os.makedirs(self.cache_config.cache_dir, exist_ok=True)
            
        if self.cache_config.policy == 'lru':
            self.frame_cache = {}
            self.cache_order = deque()
        elif self.cache_config.policy == 'lfu':
            self.frame_cache = {}
            self.cache_frequency = {}
        else:  # 'fifo'
            self.frame_cache = {}
            self.cache_queue = deque()

    def _setup_monitoring(self) -> None:
        """Setup performance monitoring."""
        if not self.export_metrics:
            return

        self.metrics_lock = threading.Lock()
        self.monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitoring_thread.start()

    def _monitor_performance(self) -> None:
        """Monitor processing performance and export metrics."""
        import psutil
        process = psutil.Process()

        while True:
            with self.metrics_lock:
                self.metrics.peak_memory_usage = max(
                    self.metrics.peak_memory_usage,
                    process.memory_info().rss
                )

                if self.export_metrics and self.metrics_path:
                    with open(self.metrics_path, 'a') as f:
                        json.dump(self.metrics.to_dict(), f)
                        f.write('\n')

            time.sleep(self.monitoring_interval)

    def _auto_tune(self) -> None:
        """Auto-tune processing parameters based on system performance."""
        if not self.auto_tune:
            return

        # Monitor system metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # Adjust workers based on CPU usage
        if cpu_percent > 90:
            self.num_workers = max(1, self.num_workers - 1)
        elif cpu_percent < 50:
            self.num_workers += 1

        # Adjust cache size based on memory usage
        if memory_percent > 90:
            self.cache_config.max_size = int(self.cache_config.max_size * 0.8)
        elif memory_percent < 50:
            self.cache_config.max_size = int(self.cache_config.max_size * 1.2)

        # Adjust streaming buffer size based on processing speed
        if self.metrics.dropped_frames / max(1, self.metrics.total_frames) > 0.1:
            self.streaming_config.buffer_size = int(self.streaming_config.buffer_size * 1.2)
        else:
            self.streaming_config.buffer_size = int(self.streaming_config.buffer_size * 0.8)
    def _update_cache(self, key: str, frame: np.ndarray) -> None:
        """Update cache based on configured policy."""
        if not self.cache_config.enabled:
            return

        with self.metrics_lock:
            if key in self.frame_cache:
                self.metrics.cache_hits += 1
            else:
                self.metrics.cache_misses += 1

        if self.cache_config.policy == 'lru':
            if key in self.frame_cache:
                self.cache_order.remove(key)
            elif len(self.frame_cache) >= self.cache_config.max_size:
                old_key = self.cache_order.popleft()
                del self.frame_cache[old_key]
            
            self.frame_cache[key] = frame
            self.cache_order.append(key)

        elif self.cache_config.policy == 'lfu':
            if key not in self.frame_cache and len(self.frame_cache) >= self.cache_config.max_size:
                min_freq_key = min(self.cache_frequency.items(), key=lambda x: x[1])[0]
                del self.frame_cache[min_freq_key]
                del self.cache_frequency[min_freq_key]
            
            self.frame_cache[key] = frame
            self.cache_frequency[key] = self.cache_frequency.get(key, 0) + 1

        else:  # 'fifo'
            if key not in self.frame_cache and len(self.frame_cache) >= self.cache_config.max_size:
                old_key = self.cache_queue.popleft()
                del self.frame_cache[old_key]
            
            self.frame_cache[key] = frame
            self.cache_queue.append(key)

        # Handle persistent caching
        if self.cache_config.persistent and self.cache_config.cache_dir:
            cache_path = Path(self.cache_config.cache_dir) / f"{key}.npy"
            if self.cache_config.compression == 'lz4':
                import lz4.frame
                with open(cache_path, 'wb') as f:
                    compressed = lz4.frame.compress(frame.tobytes())
                    f.write(compressed)
            elif self.cache_config.compression == 'zstd':
                import zstandard as zstd
                with open(cache_path, 'wb') as f:
                    cctx = zstd.ZstdCompressor(level=3)
                    compressed = cctx.compress(frame.tobytes())
                    f.write(compressed)
            else:
                np.save(cache_path, frame)

    def _process_frame_batch(self, frames: List[np.ndarray]) -> Union[np.ndarray, Any]:
        """Process a batch of frames with GPU acceleration if available."""
        if not frames:
            return frames

        if self.frame_format == FrameFormat.TORCH:
            import torch
            batch = torch.stack([torch.from_numpy(f) for f in frames])
            if self.device:
                batch = batch.to(self.device)
            return batch

        elif self.frame_format == FrameFormat.TENSORFLOW:
            import tensorflow as tf
            return tf.convert_to_tensor(frames)

        elif self.frame_format == FrameFormat.JAX:
            import jax.numpy as jnp
            return jnp.array(frames)

        elif self.frame_format == FrameFormat.PADDLE:
            import paddle
            return paddle.to_tensor(frames)

        elif self.frame_format == FrameFormat.ONNX:
            import onnx
            import onnxruntime as ort
            return np.array(frames)  # ONNX expects numpy array

        return np.array(frames)

    def _handle_streaming(self, frame: np.ndarray) -> bool:
        """Handle frame streaming based on configured mode."""
        if self.streaming_config.mode == StreamingMode.SEQUENTIAL:
            self.frame_queue.put(frame)
            return True

        elif self.streaming_config.mode == StreamingMode.ADAPTIVE:
            if self.frame_queue.qsize() / self.streaming_config.buffer_size > self.streaming_config.drop_threshold:
                with self.metrics_lock:
                    self.metrics.dropped_frames += 1
                return False
            self.frame_queue.put(frame)
            return True

        elif self.streaming_config.mode == StreamingMode.PRIORITY:
            if self.streaming_config.priority_key:
                priority = self.streaming_config.priority_key(frame)
                import heapq
                heapq.heappush(self.priority_queue, (-priority, frame))
                if len(self.priority_queue) > self.streaming_config.buffer_size:
                    heapq.heappop(self.priority_queue)
            return True

        elif self.streaming_config.mode == StreamingMode.REAL_TIME:
            try:
                self.frame_queue.put(frame, timeout=self.streaming_config.max_latency)
                return True
            except Queue.Full:
                with self.metrics_lock:
                    self.metrics.dropped_frames += 1
                return False

        return True

    def _process_video(self, metadata: VideoMetadata) -> Iterator[Union[np.ndarray, List[np.ndarray]]]:
        """Process a single video with all configured optimizations."""
        try:
            frames_buffer = []
            start_time = time.time()

            for frame in self._extract_frames(metadata):
                # Update metrics
                with self.metrics_lock:
                    self.metrics.processed_frames += 1
                    self.metrics.avg_processing_time = (
                        time.time() - start_time
                    ) / self.metrics.processed_frames

                # Handle streaming
                if not self._handle_streaming(frame):
                    continue

                # Batch processing
                if self.batch_size:
                    frames_buffer.append(frame)
                    if len(frames_buffer) >= self.batch_size:
                        yield self._process_frame_batch(frames_buffer)
                        frames_buffer = []
                else:
                    yield frame

            # Process remaining frames in buffer
            if frames_buffer:
                yield self._process_frame_batch(frames_buffer)

        except Exception as e:
            logger.error(f"Error processing video {metadata.video_id}: {str(e)}")
            raise

        finally:
            # Auto-tune parameters if enabled
            if self.auto_tune:
                self._auto_tune()
        self.frame_skip = max(1, frame_skip)
        self.target_size = target_size
        self.normalize = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.device = device
        self.augmentation = augmentation or AugmentationConfig()
        
        if seed is not None:
            np.random.seed(seed)

        # Initialize caching
        self.frame_cache = {}
        self.cache_size = cache_size
        self.cache_lock = threading.Lock()

        # Initialize queues for prefetching
        self.frame_queue = Queue(maxsize=prefetch_factor * batch_size if batch_size else 100)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        if use_temp_dir:
            self.download_dir = Path(tempfile.mkdtemp(prefix="video_dloader_"))
        else:
            self.download_dir = Path.cwd()

        # Import ML framework modules lazily
        self._setup_ml_framework()

    def _setup_ml_framework(self) -> None:
        """Lazy import of ML framework modules based on frame_format."""
        if self.frame_format == FrameFormat.TORCH:
            import torch
            self.torch = torch
        elif self.frame_format == FrameFormat.TENSORFLOW:
            import tensorflow as tf
            self.tf = tf
        elif self.frame_format == FrameFormat.PIL:
            from PIL import Image
            self.Image = Image

    def __del__(self) -> None:
        """Cleanup resources."""
        self.executor.shutdown()
        if self.use_temp_dir and self.download_dir.exists():
            shutil.rmtree(self.download_dir)

    def _apply_augmentation(self, frame: np.ndarray) -> np.ndarray:
        """Apply configured augmentations to a frame."""
        if self.augmentation.horizontal_flip and np.random.random() > 0.5:
            frame = cv2.flip(frame, 1)

        if self.augmentation.vertical_flip and np.random.random() > 0.5:
            frame = cv2.flip(frame, 0)

        if self.augmentation.rotation_range != (0, 0):
            angle = np.random.uniform(*self.augmentation.rotation_range)
            matrix = cv2.getRotationMatrix2D(
                (frame.shape[1] / 2, frame.shape[0] / 2),
                angle, 1.0
            )
            frame = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))

        if self.augmentation.brightness_range != (1.0, 1.0):
            factor = np.random.uniform(*self.augmentation.brightness_range)
            frame = cv2.convertScaleAbs(frame, alpha=factor, beta=0)

        if self.augmentation.contrast_range != (1.0, 1.0):
            factor = np.random.uniform(*self.augmentation.contrast_range)
            mean = np.mean(frame)
            frame = (frame - mean) * factor + mean

        if self.augmentation.noise_stddev > 0:
            noise = np.random.normal(0, self.augmentation.noise_stddev, frame.shape)
            frame = frame + noise

        if self.augmentation.random_crop:
            h, w = self.augmentation.random_crop
            max_x = frame.shape[1] - w
            max_y = frame.shape[0] - h
            x = np.random.randint(0, max_x + 1)
            y = np.random.randint(0, max_y + 1)
            frame = frame[y:y+h, x:x+w]

        return np.clip(frame, 0, 255).astype(np.uint8)

    def _convert_color_space(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to target color space."""
        if self.color_space == ColorSpace.RGB:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.color_space == ColorSpace.GRAYSCALE:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.color_space == ColorSpace.HSV:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return frame  # BGR

    def _convert_frame_format(self, frame: np.ndarray) -> Any:
        """Convert frame to target format."""
        if self.frame_format == FrameFormat.NUMPY:
            return frame
        elif self.frame_format == FrameFormat.TORCH:
            tensor = self.torch.from_numpy(frame)
            if self.device:
                tensor = tensor.to(self.device)
            return tensor
        elif self.frame_format == FrameFormat.TENSORFLOW:
            return self.tf.convert_to_tensor(frame)
        elif self.frame_format == FrameFormat.PIL:
            return self.Image.fromarray(frame)

    def _preprocess_frame(self, frame: np.ndarray) -> Any:
        """Complete frame preprocessing pipeline."""
        # Color space conversion
        frame = self._convert_color_space(frame)

        # Augmentation
        frame = self._apply_augmentation(frame)

        # Resize if needed
        if self.target_size:
            frame = cv2.resize(frame, self.target_size)

        # Normalize if needed
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0

        # Convert to target format
        return self._convert_frame_format(frame)

    def _prefetch_frames(self, metadata: VideoMetadata) -> None:
        """Prefetch frames in a separate thread."""
        try:
            for frame in self._extract_frames(metadata):
                self.frame_queue.put(frame)
        except Exception as e:
            self.frame_queue.put(e)
        finally:
            self.frame_queue.put(None)  # Signal completion

    def _extract_frames(self, metadata: VideoMetadata) -> Iterator[np.ndarray]:
        """Extract frames from video file."""
        try:
            local_path = metadata.local_path
            if local_path is None or not os.path.exists(local_path):
                if not metadata.remote_path:
                    raise VideoLoadError("No local or remote path available")
                
                downloaded_path = self.download_dir / f"{metadata.video_id}{Path(metadata.remote_path).suffix}"
                self._validate_video_format(str(downloaded_path))
                self.client.download(metadata.remote_path, str(downloaded_path))
                local_path = str(downloaded_path)

            cap = cv2.VideoCapture(local_path)
            if not cap.isOpened():
                raise VideoLoadError(f"Failed to open video: {local_path}")

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.frame_skip == 0:
                    # Check cache first
                    cache_key = f"{metadata.video_id}_{frame_count}"
                    with self.cache_lock:
                        if cache_key in self.frame_cache:
                            yield self.frame_cache[cache_key]
                            continue

                    # Process frame
                    processed_frame = self._preprocess_frame(frame)

                    # Update cache
                    with self.cache_lock:
                        if len(self.frame_cache) >= self.cache_size:
                            self.frame_cache.pop(next(iter(self.frame_cache)))  # Remove oldest
                        self.frame_cache[cache_key] = processed_frame

                    yield processed_frame

                frame_count += 1

        except Exception as e:
            raise VideoLoadError(f"Error processing video {metadata.video_id}: {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()

    def get_dataset_iterator(self, 
                          split: DatasetSplit,
                          transform: Optional[Callable] = None
                          ) -> Iterator[Union[np.ndarray, List[np.ndarray]]]:
        """Get an iterator for a specific dataset split."""
        videos = self.catalog.get_dataset_split(split)
        
        for video in videos:
            # Start prefetching
            future = self.executor.submit(self._prefetch_frames, video)
            
            frames_buffer = []
            while True:
                frame = self.frame_queue.get()
                if frame is None:
                    break
                if isinstance(frame, Exception):
                    raise frame

                if transform:
                    frame = transform(frame)

                if self.batch_size:
                    frames_buffer.append(frame)
                    if len(frames_buffer) == self.batch_size:
                        yield frames_buffer
                        frames_buffer = []
                else:
                    yield frame

            # Yield remaining frames
            if self.batch_size and frames_buffer:
                yield frames_buffer

            future.result()  # Wait for prefetching to complete

    def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """Get detailed information about a video."""
        metadata = self.catalog.get_video(video_id)
        if not metadata:
            raise VideoLoadError(f"Video not found: {video_id}")

        cap = cv2.VideoCapture(metadata.local_path or metadata.remote_path)
        info = {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": metadata.duration,
            "codec": metadata.codec,
            "size": metadata.file_size,
            "categories": metadata.categories,
            "split": metadata.dataset_split.value,
            "quality": metadata.quality.value
        }
        cap.release()
        return info
