def extract_valid_samples(batch_data: list, condition_func=lambda x: x != "") -> tuple[list, list]:
    """
    Extract valid samples from a batch based on a condition function.
    
    Args:
        batch_data: List of data samples
        condition_func: Function that returns True for valid samples
        
    Returns:
        Tuple of (valid_indices, valid_samples)
    """
    valid_pairs = [(i, x) for i, x in enumerate(batch_data) if condition_func(x)]
    return zip(*valid_pairs) if valid_pairs else ([], [])