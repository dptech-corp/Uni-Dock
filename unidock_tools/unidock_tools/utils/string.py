from pathlib import Path
import os
import datetime
import string
import random


def randstr(length: int = 4) -> str:
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def make_tmp_dir(
        prefix: str = "results",
        date: bool = True,
        random_id: bool = False
) -> Path:
    name = prefix
    if date:
        now = datetime.datetime.now()
        name += f"_{now.strftime('%Y%m%d%H%M%S')}"
    if random_id:
        name += f"_{randstr(8)}"
    os.makedirs(name, exist_ok=True)
    return Path(name)
