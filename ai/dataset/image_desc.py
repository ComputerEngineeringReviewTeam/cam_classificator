class ImageFields:
    """
        Class for storing keys of image description (names of fields)
        to be used for reading and writing csv files from db.
    """
    ImageId = "image_id"
    TotalArea: str = "total_area"
    TotalLength: str = "total_length"
    MeanThickness: str = "mean_thickness"
    BranchingPoints: str = "branching_points"
    IsGood: str = "is_good"
    Scale: str = "scale"
    PhotoType: str = "photo_type"
