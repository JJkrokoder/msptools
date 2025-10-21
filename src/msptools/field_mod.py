

class Field:
    """Class representing an electromagnetic field."""
    
    def __init__(self, frequency: float) -> None:
        """
        Initialize a Field object by specifying its frequency.

        Parameters:
        frequency (float): Frequency of the electromagnetic field.
        """
        self.frequency = frequency