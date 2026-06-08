import base64
import json
from typing import IO, Iterator

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_record(f: IO[str], record: dict) -> None:
    out = dict(record)
    if "vector" in out and out["vector"] is not None:
        arr = np.array(out["vector"], dtype=np.float32)
        out["vector"] = base64.b64encode(arr.tobytes()).decode("ascii")
    f.write(json.dumps(out, cls=_NumpyEncoder) + "\n")


def read_manifest(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "vector" in record and record["vector"] is not None:
                record["vector"] = base64.b64decode(record["vector"])
            yield record
