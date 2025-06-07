from typing import List
from pydantic import BaseModel


class SegmentData(BaseModel):
    """
    DTO for a single analyzed image segment.
    """
    branching_point: float|None = None
    total_length: float|None = None
    mean_thickness: float|None = None
    total_area: float|None =None
    is_good: bool|None = None


class ClassificationResult(BaseModel):
    """
    The complete DTO returned by the classification API.
    """
    segment_width: int
    segment_height: int
    segments: List[SegmentData]
    num_cols: int = None
    num_rows: int = None
    segments: List[SegmentData]