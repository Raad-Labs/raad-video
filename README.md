<div align="center">
  <img src="https://pbs.twimg.com/profile_banners/5385912/1737222476/1500x500" alt="RAAD Video Banner" width="100%">
</div>

# RAAD Video

A high-performance video loading library for machine learning, designed for efficient training data preparation.

## Features

- **High-Performance Processing**
  - Fast video frame extraction and preprocessing
  - Multi-threaded and distributed processing support
  - Smart caching with Redis for improved throughput
  - Memory-efficient streaming with adaptive buffering

- **ML Framework Integration**
  - Native support for PyTorch, TensorFlow, JAX, and more
  - Optimized tensor conversions and memory management
  - GPU acceleration support
  - Configurable output formats and color spaces

- **Advanced Capabilities**
  - Sophisticated augmentation pipeline
  - Real-time performance monitoring
  - Auto-tuning for optimal performance
  - Peer-to-peer sharing for distributed setups

## Installation

RAAD Video requires Python 3.8 or later. Install via pip:

```bash
pip install raad-video
```

For development installation with testing and code quality tools:
```bash
pip install "raad-video[dev]"
```

### System Requirements
- Python 3.8+
- OpenCV dependencies
- Redis (optional, for distributed caching)
- CUDA-compatible GPU (optional, for GPU acceleration)

## Quick Start

```python
from raad import VideoDataLoader, VideoCatalog
from raad.config import ProcessingMode, StreamingConfig, FrameFormat

# Create a catalog of your videos
catalog = VideoCatalog()
catalog.add_video("video1.mp4", categories=["training"])
catalog.add_video("video2.mp4", categories=["validation"])

# Initialize the loader with optimal settings
loader = VideoDataLoader(
    catalog=catalog,
    processing_mode=ProcessingMode.MULTI_THREAD,
    frame_format=FrameFormat.TORCH,  # Output PyTorch tensors
    streaming_config=StreamingConfig(
        mode="adaptive",
        buffer_size=1000
    ),
    target_size=(224, 224),  # Resize frames
    normalize=True,          # Normalize pixel values
    device="cuda"          # Use GPU if available
)

# Get frames for training
for frames in loader.get_dataset_iterator("training"):
    # frames will be preprocessed and ready for your model
    # Shape: (batch_size, channels, height, width)
    model.train(frames)
```

## Advanced Usage

### Distributed Processing

```python
from raad.config import DistributedConfig

# Setup distributed processing across multiple nodes
loader = VideoDataLoader(
    catalog=catalog,
    processing_mode=ProcessingMode.DISTRIBUTED,
    distributed_config=DistributedConfig(
        enabled=True,
        num_nodes=4,
        node_rank=0,
        master_addr='10.0.0.1',
        master_port=29500
    )
)
```

### Custom Augmentation Pipeline

```python
from raad.augmentation import (
    RandomBrightness,
    RandomContrast,
    RandomFlip,
    RandomRotation,
    ColorJitter
)

# Create a sophisticated augmentation pipeline
loader = VideoDataLoader(
    catalog=catalog,
    augmentations=[
        RandomBrightness(0.2),
        RandomContrast(0.2),
        RandomFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
    ]
)
```

### Performance Optimization

```python
from raad.config import CacheConfig, StreamingConfig

# Configure caching and streaming for optimal performance
loader = VideoDataLoader(
    catalog=catalog,
    cache_config=CacheConfig(
        enabled=True,
        policy="lru",
        max_size_gb=100,
        persistent=True,
        compression="lz4"
    ),
    streaming_config=StreamingConfig(
        mode="adaptive",
        buffer_size=1000,
        max_latency=0.1,
        drop_threshold=0.8
    ),
    auto_tune=True,  # Enable automatic performance tuning
    monitoring_interval=1.0,  # Monitor performance every second
    export_metrics=True
)
```

## Troubleshooting

### Common Issues

1. **Memory Usage**
   - Use `streaming_config` with appropriate `buffer_size`
   - Enable frame dropping with `drop_threshold` if needed
   - Consider using persistent caching

2. **Performance**
   - Enable `auto_tune` for automatic optimization
   - Use appropriate `processing_mode` for your setup
   - Monitor performance with `export_metrics=True`

3. **GPU Issues**
   - Ensure CUDA is properly installed
   - Set appropriate `device` and `batch_size`
   - Monitor GPU memory usage

### Getting Help

- Open an issue on GitHub
- Check the API documentation
- Join our community on Discord
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
