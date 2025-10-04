import re
from unicodedata import normalize


def slugify(value: str, /, allow_unicode: bool = False) -> str:
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = normalize("NFKC", value)
    else:
        value = normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    value = re.sub(r"[-\s]+", "_", value).strip("-_")
    return value


__all__ = [
    "slugify",
]
