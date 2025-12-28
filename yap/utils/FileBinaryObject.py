class FileBinaryObject:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None

    def open(self):
        """Open the file with the given mode."""
        self.file = open(self.filepath, 'rb+')

    def readFull(self):
        """Read the contents of the file."""
        if self.file is None:
            raise ValueError("File is not open.")
        return self.file.read()

    def readNBytes(self, n):
        """Read n bytes from the file."""
        if self.file is None:
            raise ValueError("File is not open.")
        return self.file.read(n)

    def reset(self):
        """Reset the file pointer to the beginning."""
        if self.file is None:
            raise ValueError("File is not open.")
        self.file.seek(0)
        
    def close(self):
        """Close the file."""
        if self.file is not None:
            self.file.close()
            self.file = None