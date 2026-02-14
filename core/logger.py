import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class Logger:
    def __init__(
        self,
        log_root: Path = Path(""),
        name: str | None = None,
        metadata: dict | None = None,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{name}" if name else ""

        self.log_root = Path(log_root)
        self.path = self.log_root / "logs" / f"{prefix}{timestamp}_raw.ndjson"
        self.path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {"metadata": None} if metadata is None else {"metadata": metadata}

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    def log_event(self, metrics: dict, meta: dict | None = None):
        record = {"timestamp": datetime.now(timezone.utc).isoformat()}

        if meta is not None:
            record.update(meta)

        record["metrics"] = metrics

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read_raw_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _build_yaml_payload(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {}

        first = records[0]
        metadata = first.get("metadata") if isinstance(first, dict) else None
        metadata = metadata if isinstance(metadata, dict) else {}
        events = records[1:]

        yaml_stem = self.path.stem
        if yaml_stem.endswith("_raw"):
            yaml_stem = yaml_stem[:-4]

        root = dict(metadata)
        root["metrics"] = events
        return {yaml_stem: root}

    def _sanity_check_conversion(self, records: list[dict[str, Any]], payload: dict[str, Any]) -> bool:
        if not records or not payload or len(payload) != 1:
            return False

        doc_name = next(iter(payload.keys()))
        doc = payload.get(doc_name)
        if not isinstance(doc, dict):
            return False

        source_meta = records[0].get("metadata") if isinstance(records[0], dict) else None
        source_meta = source_meta if isinstance(source_meta, dict) else {}
        source_events = records[1:]

        # Ensure all metadata cells are preserved.
        for key, value in source_meta.items():
            if key not in doc or doc[key] != value:
                return False

        # Ensure all event cells are preserved.
        metrics = doc.get("metrics")
        if not isinstance(metrics, list) or len(metrics) != len(source_events):
            return False

        for src, dst in zip(source_events, metrics):
            if src != dst:
                return False

        return True

    def finalize(self) -> Path | None:
        records = self._read_raw_records()
        payload = self._build_yaml_payload(records)

        if not self._sanity_check_conversion(records, payload):
            print(f"[logger] conversion sanity-check failed, keeping raw log: {self.path}")
            return None

        yaml_stem = self.path.stem
        if yaml_stem.endswith("_raw"):
            yaml_stem = yaml_stem[:-4]

        yaml_path = self.path.with_name(f"{yaml_stem}.yaml")
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

        self.path.unlink(missing_ok=True)
        return yaml_path
