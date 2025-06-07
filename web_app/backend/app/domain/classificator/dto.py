from typing import List
from pydantic import BaseModel


class SegmentData(BaseModel):
    """
    DTO for a single analyzed image segment.
    """
    branching_point: int|None = None
    is_good: bool|None = None


class ClassificationResult(BaseModel):
    """
    The complete DTO returned by the classification API.
    """
    branching_point_sum: int|None = None
    is_good_percent: float|None = None
    segment_width: int
    segment_height: int
    overflowed_segment_width: int
    overflowed_segment_height: int
    segments: List[SegmentData]
    num_cols: int = None
    num_rows: int = None
    segments: List[SegmentData]