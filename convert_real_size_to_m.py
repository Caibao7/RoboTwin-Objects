#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

def convert_real_size_to_m(input_file: str, output_file: str = None):
    input_path = Path(input_file)
    if output_file is None:
        output_file = input_path.with_name(input_path.stem + "_meter.json")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = {}
    for uuid, obj in data.items():
        real_size_str = obj.get("real_size", "")
        try:
            dims_cm = [float(x) for x in real_size_str.split("*")]
            dims_m = [round(x / 100.0, 4) for x in dims_cm]  # 保留4位小数
            obj["real_size"] = "*".join(str(x) for x in dims_m)
        except Exception as e:
            print(f"[WARN] Failed to parse real_size for {uuid}: {real_size_str} ({e})")
        new_data[uuid] = obj

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"Converted file saved to {output_file}")

if __name__ == "__main__":
    # 用法： python convert_real_size.py robotwin_real_sizes.json
    import sys
    if len(sys.argv) < 2:
        print("Usage: python convert_real_size.py <input_json> [output_json]")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        convert_real_size_to_m(input_file, output_file)
