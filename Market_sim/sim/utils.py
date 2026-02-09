"""
Mathematical utility functions for market simulation.

Provides helper functions for common mathematical operations.
"""


class MathX:
    """Collection of mathematical utility functions."""
    
    @staticmethod
    def round_to_int(x: float) -> int:
        """
        Round float to nearest integer.
        
        Args:
            x: Float to round
            
        Returns:
            Rounded integer value
        """
        return int(round(x))

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        """
        Clamp value to range [lo, hi].
        
        Args:
            x: Value to clamp
            lo: Lower bound
            hi: Upper bound
            
        Returns:
            Clamped value
        """
        return max(lo, min(hi, x))

    @staticmethod
    def max_int(a: int, b: int) -> int:
        """
        Return maximum of two integers.
        
        Args:
            a: First integer
            b: Second integer
            
        Returns:
            Maximum value
        """
        return a if a >= b else b

    @staticmethod
    def min_int(a: int, b: int) -> int:
        """
        Return minimum of two integers.
        
        Args:
            a: First integer
            b: Second integer
            
        Returns:
            Minimum value
        """
        return a if a <= b else b
