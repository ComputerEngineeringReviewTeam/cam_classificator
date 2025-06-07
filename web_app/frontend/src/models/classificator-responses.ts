export interface SegmentData {
  branching_point: number | null;
  is_good: boolean | null;
}

export interface ClassificationResult {
  branching_point_sum: number;
  is_good_percent: number;
  segment_width: number;
  segment_height: number;
  overflowed_segment_width: number;
  overflowed_segment_height: number;
  num_cols: number;
  num_rows: number;
  segments: SegmentData[];
}