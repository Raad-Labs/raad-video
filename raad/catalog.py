# catalog.py
import os
from typing import List, Dict, Any

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from enum import Enum
from datetime import datetime

class DatasetSplit(Enum):
    """Dataset split for ML training."""
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"

class VideoQuality(Enum):
    """Video quality/resolution categories."""
    LOW = "low"      # e.g., 480p
    MEDIUM = "med"   # e.g., 720p
    HIGH = "high"    # e.g., 1080p
    ULTRA = "ultra"  # e.g., 4K+

@dataclass
class VideoMetadata:
    """
    Enhanced metadata for AI/ML video processing:
    - Basic identification and paths
    - ML-specific attributes (split, quality, etc.)
    - Technical metadata (codec, fps, etc.)
    - Content metadata (scene info, timestamps, etc.)
    - Training metadata (embeddings, features, etc.)
    """
    # Basic identification
    video_id: str
    categories: List[str]
    local_path: Optional[str] = None
    remote_path: Optional[str] = None
    
    # ML-specific attributes
    dataset_split: DatasetSplit = DatasetSplit.TRAIN
    quality: VideoQuality = VideoQuality.HIGH
    weight: float = 1.0  # Sample weight for training
    
    # Technical metadata
    duration: Optional[float] = None  # in seconds
    fps: Optional[float] = None
    resolution: Optional[tuple[int, int]] = None  # (width, height)
    codec: Optional[str] = None
    file_size: Optional[int] = None  # in bytes
    
    # Content metadata
    scene_timestamps: List[Dict[str, Any]] = field(default_factory=list)  # List of scene changes/cuts
    keyframes: List[float] = field(default_factory=list)  # Timestamp of important frames
    annotations: Dict[str, Any] = field(default_factory=dict)  # Manual or auto annotations
    
    # Training metadata
    embeddings_path: Optional[str] = None  # Path to pre-computed embeddings
    features_path: Optional[str] = None    # Path to extracted features
    augmentation_history: List[str] = field(default_factory=list)  # Applied augmentations
    
    # Tracking and validation
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    checksum: Optional[str] = None  # For data integrity
    version: str = "1.0"  # Schema version
    
    def __post_init__(self):
        """Validate and process after initialization."""
        if not self.categories:
            raise ValueError("Video must have at least one category")
        
        # Convert lists to sets for faster lookup
        self._categories_set = set(self.categories)
        
    def has_category(self, category: str) -> bool:
        """Efficient category membership check."""
        return category in self._categories_set
    
    def add_scene(self, timestamp: float, scene_type: str, confidence: float = 1.0, **metadata) -> None:
        """Add a scene timestamp with metadata."""
        self.scene_timestamps.append({
            "timestamp": timestamp,
            "type": scene_type,
            "confidence": confidence,
            **metadata
        })
    
    def add_keyframe(self, timestamp: float) -> None:
        """Add a keyframe timestamp."""
        self.keyframes.append(timestamp)
        self.keyframes.sort()
    
    def update_access_time(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed = datetime.now()
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get all training-related metadata."""
        return {
            "split": self.dataset_split.value,
            "weight": self.weight,
            "embeddings": self.embeddings_path,
            "features": self.features_path,
            "augmentations": self.augmentation_history
        }
    
    def get_technical_info(self) -> Dict[str, Any]:
        """Get all technical metadata."""
        return {
            "duration": self.duration,
            "fps": self.fps,
            "resolution": self.resolution,
            "codec": self.codec,
            "file_size": self.file_size
        }

    def __repr__(self) -> str:
        return (f"<VideoMetadata id={self.video_id}, "
                f"categories={self.categories}, "
                f"split={self.dataset_split.value}, "
                f"quality={self.quality.value}>")

from collections import defaultdict
import json
from pathlib import Path
from typing import Iterator, Callable, TypeVar, Generic

T = TypeVar('T')

class DatasetIterator(Generic[T]):
    """Iterator for dataset splits with optional transforms."""
    def __init__(self, items: List[T], transform: Optional[Callable[[T], T]] = None):
        self.items = items
        self.transform = transform
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self) -> T:
        if self.index >= len(self.items):
            raise StopIteration
        item = self.items[self.index]
        self.index += 1
        if self.transform:
            item = self.transform(item)
        return item

class VideoCatalog:
    """
    Enhanced catalog for ML-focused video management.
    Supports advanced querying, dataset splits, and ML metadata tracking.
    """
    def __init__(self, name: str = "default") -> None:
        self._videos: Dict[str, VideoMetadata] = {}
        self._category_index: Dict[str, Set[str]] = defaultdict(set)
        self._split_index: Dict[DatasetSplit, Set[str]] = defaultdict(set)
        self._quality_index: Dict[VideoQuality, Set[str]] = defaultdict(set)
        self.name = name

    def add_video(self, metadata: VideoMetadata) -> None:
        """Add a video with indexing for fast querying."""
        if metadata.video_id in self._videos:
            raise ValueError(f"Video with ID {metadata.video_id} already exists")

        # Update indices
        for category in metadata.categories:
            self._category_index[category].add(metadata.video_id)
        self._split_index[metadata.dataset_split].add(metadata.video_id)
        self._quality_index[metadata.quality].add(metadata.video_id)

        metadata.update_access_time()
        self._videos[metadata.video_id] = metadata

    def update_video(self, metadata: VideoMetadata) -> None:
        """Update video with re-indexing."""
        if metadata.video_id not in self._videos:
            raise KeyError(f"Video with ID {metadata.video_id} not found")

        old_metadata = self._videos[metadata.video_id]

        # Remove old indices
        for category in old_metadata.categories:
            self._category_index[category].discard(metadata.video_id)
        self._split_index[old_metadata.dataset_split].discard(metadata.video_id)
        self._quality_index[old_metadata.quality].discard(metadata.video_id)

        # Add new indices
        for category in metadata.categories:
            self._category_index[category].add(metadata.video_id)
        self._split_index[metadata.dataset_split].add(metadata.video_id)
        self._quality_index[metadata.quality].add(metadata.video_id)

        metadata.update_access_time()
        self._videos[metadata.video_id] = metadata

    def get_dataset_split(self, split: DatasetSplit, transform: Optional[Callable[[VideoMetadata], VideoMetadata]] = None) -> DatasetIterator[VideoMetadata]:
        """Get an iterator for a specific dataset split with optional transform."""
        videos = [self._videos[vid] for vid in self._split_index[split]]
        return DatasetIterator(videos, transform)

    def get_videos_by_quality(self, quality: VideoQuality) -> List[VideoMetadata]:
        """Get all videos of a specific quality."""
        return [self._videos[vid] for vid in self._quality_index[quality]]

    def filter_videos(self, **criteria) -> List[VideoMetadata]:
        """Filter videos by multiple criteria."""
        results = set(self._videos.keys())

        if 'categories' in criteria:
            category_videos = set.intersection(*[self._category_index[cat] 
                                               for cat in criteria['categories']])
            results &= category_videos

        if 'split' in criteria:
            results &= self._split_index[criteria['split']]

        if 'quality' in criteria:
            results &= self._quality_index[criteria['quality']]

        if 'min_duration' in criteria:
            results &= {vid for vid in results 
                       if self._videos[vid].duration >= criteria['min_duration']}

        if 'max_duration' in criteria:
            results &= {vid for vid in results 
                       if self._videos[vid].duration <= criteria['max_duration']}

        return [self._videos[vid] for vid in results]

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        stats = {
            "total_videos": len(self._videos),
            "total_duration": sum(v.duration or 0 for v in self._videos.values()),
            "splits": {split.value: len(videos) for split, videos in self._split_index.items()},
            "qualities": {quality.value: len(videos) for quality, videos in self._quality_index.items()},
            "categories": {cat: len(videos) for cat, videos in self._category_index.items()}
        }
        return stats

    def save_to_disk(self, path: str) -> None:
        """Save catalog to disk in JSON format."""
        data = {
            "name": self.name,
            "videos": {vid: {
                **vars(metadata),
                "dataset_split": metadata.dataset_split.value,
                "quality": metadata.quality.value,
                "created_at": metadata.created_at.isoformat(),
                "last_accessed": metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                "last_modified": metadata.last_modified.isoformat() if metadata.last_modified else None
            } for vid, metadata in self._videos.items()}
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_disk(cls, path: str) -> 'VideoCatalog':
        """Load catalog from disk."""
        with open(path, 'r') as f:
            data = json.load(f)

        catalog = cls(name=data['name'])
        for vid, video_data in data['videos'].items():
            # Convert string values back to enums
            video_data['dataset_split'] = DatasetSplit(video_data['dataset_split'])
            video_data['quality'] = VideoQuality(video_data['quality'])
            
            # Parse datetime strings
            video_data['created_at'] = datetime.fromisoformat(video_data['created_at'])
            if video_data['last_accessed']:
                video_data['last_accessed'] = datetime.fromisoformat(video_data['last_accessed'])
            if video_data['last_modified']:
                video_data['last_modified'] = datetime.fromisoformat(video_data['last_modified'])

            metadata = VideoMetadata(**video_data)
            catalog.add_video(metadata)

        return catalog