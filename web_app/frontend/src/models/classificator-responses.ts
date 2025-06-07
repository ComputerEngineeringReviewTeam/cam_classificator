export interface SegmentData {
  branching_point: number | null;
  total_length: number | null;
  mean_thickness: number | null;
  total_area: number | null;
  is_good: boolean | null;
}

export interface ClassificationResult {
  segment_width: number;
  segment_height: number;
  num_cols: number;
  num_rows: number;
  segments: SegmentData[];
}