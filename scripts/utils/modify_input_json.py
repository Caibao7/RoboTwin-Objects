#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
import sys
import os


def load_map(path):
    """
    读取映射 CSV，返回 {uuid: (source_dir, index)}。
    兼容带/不带表头、UTF-8 含 BOM、以及前 3 列分别为 uuid/source_dir/index 的情况。
    """
    mapping = {}
    if not os.path.exists(path):
        return mapping

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        try:
            reader = csv.DictReader(f)
            if reader.fieldnames and {"uuid", "source_dir", "index"} <= {h.strip().lower() for h in reader.fieldnames}:
                # 标准表头
                for row in reader:
                    if not row:
                        continue
                    u = (row.get("uuid") or "").strip()
                    if not u:
                        continue
                    sd = (row.get("source_dir") or "").strip()
                    idx = str(row.get("index") or "").strip()
                    mapping[u] = (sd, idx)
            else:
                # 无表头或非常规表头，按前三列
                f.seek(0)
                reader = csv.reader(f)
                first = next(reader, None)
                pos_u, pos_sd, pos_idx = 0, 1, 2
                # 如果第一行像表头且含 uuid/source_dir/index，就自动定位列号
                if first and any(isinstance(c, str) and c.strip().lower() == "uuid" for c in first):
                    lower = [c.strip().lower() if isinstance(c, str) else "" for c in first]
                    if "uuid" in lower:
                        pos_u = lower.index("uuid")
                    if "source_dir" in lower:
                        pos_sd = lower.index("source_dir")
                    if "index" in lower:
                        pos_idx = lower.index("index")
                else:
                    # 第一行就是数据，需要先吃回去
                    if first and len(first) >= 3:
                        u = (first[pos_u] or "").strip()
                        sd = (first[pos_sd] or "").strip()
                        idx = str(first[pos_idx] or "").strip()
                        if u and u.lower() != "uuid":
                            mapping[u] = (sd, idx)

                for row in reader:
                    if not row or len(row) < 3:
                        continue
                    u = (row[pos_u] or "").strip()
                    if not u or u.lower() == "uuid":
                        continue
                    sd = (row[pos_sd] or "").strip()
                    idx = str(row[pos_idx] or "").strip()
                    mapping[u] = (sd, idx)
        except Exception:
            # 极端情况下的兜底读取
            f.seek(0)
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 3:
                    continue
                u, sd, idx = (row[0] or "").strip(), (row[1] or "").strip(), str(row[2] or "").strip()
                if not u or u.lower() == "uuid":
                    continue
                mapping[u] = (sd, idx)
    return mapping


def transform(
    input_json="filtered_robotwin_dim_img.json",
    map_csv="robotwin_uuid_map.csv",
    special_csv="robotwin_uuid_map_special.csv",
    output_json="filtered_robotwin_img_with_name.json",
):
    # 1) 读取原 JSON
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2) 读取两个映射表
    primary = load_map(map_csv)
    special = load_map(special_csv)

    # 3) 按优先级生成 robotwin_name，并去掉 dimension
    new_data = {}
    stats = {"primary": 0, "special": 0, "missing": 0}

    for uuid, item in data.items():
        robotwin_name = None
        if uuid in primary:
            sd, idx = primary[uuid]
            robotwin_name = f"{sd}_{idx}"
            stats["primary"] += 1
        elif uuid in special:
            sd, idx = special[uuid]
            robotwin_name = f"{sd}_{idx}"
            stats["special"] += 1
        else:
            stats["missing"] += 1

        new_data[uuid] = {
            "image": item.get("image"),
            "robotwin_name": robotwin_name,  # 找不到则为 None -> JSON 中是 null
        }

    # 4) 写出新 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    # 简单统计输出
    print(
        f"完成。总数: {len(data)} | 命中主表: {stats['primary']} | 命中特表: {stats['special']} | 未命中: {stats['missing']}"
    )
    print(f"输出文件: {os.path.abspath(output_json)}")


def main():
    # 命令行参数（可选）：input_json map_csv special_csv output_json
    args = sys.argv[1:]
    input_json = args[0] if len(args) > 0 else "filtered_robotwin_dim_img.json"
    map_csv = args[1] if len(args) > 1 else "robotwin_uuid_map.csv"
    special_csv = args[2] if len(args) > 2 else "robotwin_uuid_map_special.csv"
    output_json = args[3] if len(args) > 3 else "filtered_robotwin_img_with_name.json"

    if input_json in ("-h", "--help"):
        print(
            "用法：python script.py [input_json] [map_csv] [special_csv] [output_json]\n"
            "默认：filtered_robotwin_dim_img.json robotwin_uuid_map.csv "
            "robotwin_uuid_map_special.csv filtered_robotwin_img_with_name.json"
        )
        sys.exit(0)

    transform(input_json, map_csv, special_csv, output_json)


if __name__ == "__main__":
    main()
