from functools import lru_cache

from .slugify import slugify


@lru_cache(maxsize=1024)
def cached_slugify(value: str, allow_unicode: bool = False) -> str:
    return slugify(value, allow_unicode=allow_unicode)


__all__ = [
    "cached_slugify",
]
