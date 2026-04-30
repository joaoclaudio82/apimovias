from pathlib import Path

class PathUtils:
    @staticmethod
    def is_empty(path: Path) -> bool:
        return not path.exists() or path.stat().st_size == 0
