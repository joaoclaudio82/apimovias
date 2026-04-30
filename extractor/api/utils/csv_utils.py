import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import pandas as pd
from api.utils.path_utils import PathUtils

class CsvUtils:
    @staticmethod
    @contextmanager
    def lock(path: Path, poll_interval: float = 0.05) -> Iterator[None]:
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = None
        try:
            while True:
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    break
                except FileExistsError:
                    time.sleep(poll_interval)
            yield
        finally:
            if fd is not None:
                os.close(fd)
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

    @classmethod
    def atomic_append_dataframe(cls, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=f"{path.name}.", dir=str(path.parent))
        os.close(tmp_fd)
        tmp = Path(tmp_path)

        try:
            with cls.lock(path):
                is_empty = PathUtils.is_empty(path)
                if not is_empty and path.exists():
                    shutil.copyfile(path, tmp)

                mode = "w" if is_empty else "a"
                with open(tmp, mode, newline="") as f:
                    if not is_empty:
                        with open(tmp, "rb") as fb:
                            fb.seek(0, os.SEEK_END)
                            if fb.tell() > 0:
                                fb.seek(-1, os.SEEK_END)
                                if fb.read(1) not in (b"\n", b"\r"):
                                    f.write("\n")
                    df.to_csv(f, index=False, header=is_empty)
                    f.flush()
                    os.fsync(f.fileno())

                os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
