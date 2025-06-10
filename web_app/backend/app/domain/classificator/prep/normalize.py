def normalize_minmax(tensor, amax, amin, new_max=1.0, new_min=0.0):
    # amin, amax = torch.amin(column_tensor), torch.amax(column_tensor)
    return ((tensor - amin) / (amax - amin)) * (new_max - new_min) + new_min