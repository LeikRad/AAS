def bin2int(b):
    """Convert a binary string to an integer.

    Args:
        b (bytes): The binary string to convert.

    Returns:
        int: The integer representation of the binary string.
    """        
    return int.from_bytes(b, byteorder='big')