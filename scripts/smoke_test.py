from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.parsers import parse_message


def main() -> None:
    samples_path = Path(__file__).parent.parent / "data/samples/sample_flows.json"
    with open(samples_path) as f:
        messages = json.load(f)

    print(f"Loaded {len(messages)} sample flows\n")
    ok = 0
    for msg in messages:
        record = parse_message(msg)
        if record is None:
            print(f"  WARN: Could not parse source={msg.get('source')}")
            continue
        print(
            f"  [{record.traffic_source:8s}] flow={record.flow_id[:10]} "
            f"dur={record.duration:.2f}s "
            f"bytes={record.bytes_fwd + record.bytes_bwd:>8d} "
            f"pkts={record.packets_fwd + record.packets_bwd:>5d} "
            f"syn={record.syn_flag_cnt:>4d}"
        )
        ok += 1

    print(f"\nParsed {ok}/{len(messages)} flows successfully.")
    sys.exit(0 if ok == len(messages) else 1)


if __name__ == "__main__":
    main()
