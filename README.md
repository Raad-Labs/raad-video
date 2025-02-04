# RAAD Video

A high-performance video loading library for machine learning, designed for efficient training data preparation.

## Features

- Fast video frame extraction and preprocessing
- Distributed processing support
- Smart caching with Redis
- Peer-to-peer sharing capabilities
- Multiple ML framework support (PyTorch, TensorFlow, JAX)
- Streaming optimization
- Advanced augmentation pipeline

## Installation

```bash
pip install raad-video
```

For development installation with all extras:
```bash
pip install "raad-video[dev]"
```

## Quick Start

```python
from raad import VideoDataLoader, VideoCatalog
from raad.config import ProcessingMode, StreamingConfig

# Create a catalog of your videos
catalog = VideoCatalog()
catalog.add_video("video1.mp4", categories=["training"])
catalog.add_video("video2.mp4", categories=["validation"])

# Initialize the loader with optimal settings
loader = VideoDataLoader(
    catalog=catalog,
    processing_mode=ProcessingMode.MULTI_THREAD,
    streaming_config=StreamingConfig(
        mode="adaptive",
        buffer_size=1000
    )
)

# Get frames for training
for frames in loader.get_dataset_iterator("training"):
    # frames will be preprocessed and ready for your model
    model.train(frames)
```

## Advanced Usage

### Distributed Processing

```python
from raad.config import DistributedConfig

loader = VideoDataLoader(
    catalog=catalog,
    processing_mode=ProcessingMode.DISTRIBUTED,
    distributed_config=DistributedConfig(
        num_nodes=4,
        node_rank=0
    )
)
```

### Custom Augmentation Pipeline

```python
from raad.augmentation import (
    RandomBrightness,
    RandomContrast,
    RandomFlip
)

loader = VideoDataLoader(
    catalog=catalog,
    augmentations=[
        RandomBrightness(0.2),
        RandomContrast(0.2),
        RandomFlip(p=0.5)
    ]
)
```

### Caching Configuration

```python
from raad.config import CacheConfig

loader = VideoDataLoader(
    catalog=catalog,
    cache_config=CacheConfig(
        enabled=True,
        policy="lru",
        max_size_gb=100
    )
)
```

## Performance Tips

1. **Streaming Mode Selection**:
   - Use `ADAPTIVE` for general training
   - Use `REAL_TIME` for time-critical applications
   - Use `PRIORITY` when certain frames are more important

2. **Caching Strategy**:
   - Enable Redis for distributed setups
   - Use local caching for single machine training
   - Configure cache size based on available RAM

3. **Processing Mode**:
   - `MULTI_THREAD` for I/O-bound workloads
   - `MULTI_PROCESS` for CPU-bound preprocessing
   - `DISTRIBUTED` for large-scale training

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
