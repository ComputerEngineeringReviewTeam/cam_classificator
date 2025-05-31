import torch

def normalize_minmax(column_tensor, new_max = 1.0, new_min = 0.0):
    amin, amax = torch.amin(column_tensor), torch.amax(column_tensor)
    normalized_column_tensor = ((column_tensor - amin) / (amax - amin)) * (new_max - new_min) + new_min
    return normalized_column_tensor