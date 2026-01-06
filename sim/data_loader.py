# CSV/data loader
import csv
from datetime import datetime
from typing import List

from sim.types import HistoricalPoint

def load_cre_csv(path: str) -> List[HistoricalPoint]:
    """
    CSV format: date,price
    date can be YYYY-MM or YYYY-MM-DD.
    """
    points: List[HistoricalPoint] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # header (optional)

        for row in reader:
            if len(row) < 2:
                continue

            date_str = row[0].strip()
            price_str = row[1].strip()

            try:
                price = float(price_str)
            except ValueError:
                continue

            dt = None
            for fmt in ("%Y-%m-%d", "%Y-%m"):
                try:
                    dt = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    pass

            if dt is None:
                continue

            points.append(HistoricalPoint(dt, price))

    points.sort(key=lambda p: p.date)
    return points
