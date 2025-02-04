from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List

class ProcessingMode(str, Enum):
    SINGLE_THREAD = "single_thread"
    MULTI_THREAD = "multi_thread"
    MULTI_PROCESS = "multi_process"
    DISTRIBUTED = "distributed"

class FrameFormat(str, Enum):
    NUMPY = "numpy"
    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    PIL = "pil"

class ColorSpace(str, Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    HSV = "hsv"

class StreamingMode(str, Enum):
    SEQUENTIAL = "sequential"
    ADAPTIVE = "adaptive"
    PRIORITY = "priority"
    REAL_TIME = "real_time"

@dataclass
class StreamingConfig:
    mode: StreamingMode = StreamingMode.ADAPTIVE
    buffer_size: int = 1000
    max_latency: Optional[float] = None
    prefetch_factor: int = 2

@dataclass
class CacheConfig:
    enabled: bool = True
    policy: str = "lru"
    max_size_gb: float = 100
    compression: bool = False
    persistent: bool = True

@dataclass
class DistributedConfig:
    enabled: bool = False
    num_nodes: int = 1
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500
