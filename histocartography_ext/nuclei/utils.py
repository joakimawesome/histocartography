from typing import Generator, Tuple


def tile_iterator(
    slide_width: int,
    slide_height: int,
    tile_size: int = 256,
    overlap: int = 0,
) -> Generator[Tuple[int, int, int, int], None, None]:
    """
    Yields (x, y, w, h) for tiles covering the entire slide.

    Args:
        slide_width: Width of the slide in pixels.
        slide_height: Height of the slide in pixels.
        tile_size: Tile dimension (square).
        overlap: Overlap between adjacent tiles in pixels.
    """
    step = tile_size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than tile_size ({tile_size})")
    for y in range(0, slide_height, step):
        for x in range(0, slide_width, step):
            w = min(tile_size, slide_width - x)
            h = min(tile_size, slide_height - y)
            yield x, y, w, h
