from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class BBox:
    x1: int | None
    y1: int | None
    x2: int | None
    y2: int | None

    def as_tuple(self):
        return (self.x1, self.y1), (self.x2, self.y2)

    def normalize_coord(self, coord, size) -> int:
        match coord:
            case None:
                return size
            case float():
                return int(size * coord)
            case tuple():
                return sum(self.normalize_coord(c, size) for c in coord)
            case _ if coord < 0:
                return size + coord
            case _:
                return coord

    def normalised(self, size_x, size_y):
        (x1, y1), (x2, y2) = self.as_tuple()
        x1 = self.normalize_coord(x1, size_x)
        x2 = self.normalize_coord(x2, size_x)
        y1 = self.normalize_coord(y1, size_y)
        y2 = self.normalize_coord(y2, size_y)
        return BBox(x1, y1, x2, y2)

    def inside(self, x, y, size_x, size_y):
        (x1, y1), (x2, y2) = self.normalised(size_x, size_y).as_tuple()
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        x1, x2 = min_x, max_x
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        y1, y2 = min_y, max_y
        return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class Zone:
    bbox: BBox
    action: Callable
    colour: tuple = (255, 255, 255)
