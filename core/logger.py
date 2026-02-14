import json
from pathlib import Path
from datetime import datetime, timezone

class Logger():
    
    def __init__(
        self,
        log_root: Path = Path(""),
        name: str = None,
        metadata: dict = None
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{name}" if name else "" 

        self.log_root = log_root
        self.path = Path("logs") / f"{prefix}{timestamp}_raw.ndjson"

        self.path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {"metadata": None} if not metadata else metadata

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    def log_event(self, metrics: dict, meta: dict = None):
        record = {"timestamp": datetime.now(timezone.utc).isoformat()}

        if meta is not None:
            record.update(meta)

        record["metrics"] = metrics

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")