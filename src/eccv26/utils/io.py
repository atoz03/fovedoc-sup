from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)


def append_jsonl(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(obj), ensure_ascii=False) + "\n")


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def safe_getenv(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj

