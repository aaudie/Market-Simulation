#Math helpers
class MathX:
    @staticmethod
    def round_to_int(x: float) -> int:
        return int(round(x))

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def max_int(a: int, b: int) -> int:
        return a if a >= b else b

    @staticmethod
    def min_int(a: int, b: int) -> int:
        return a if a <= b else b
