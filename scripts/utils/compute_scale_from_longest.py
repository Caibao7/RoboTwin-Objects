# -*- coding: utf-8 -*-
import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from decimal import Decimal, ROUND_DOWN, InvalidOperation, getcontext

# 设置足够的计算精度
getcontext().prec = 28

def truncate_to_4(x: float) -> float:
    """
    将数值向零截断到最多 4 位小数（不四舍五入）。
    例如：1.23456 -> 1.2345,  -1.23456 -> -1.2345
    """
    try:
        d = Decimal(str(x))
        q = Decimal("0.0001")
        return float(d.quantize(q, rounding=ROUND_DOWN))
    except (InvalidOperation, ValueError):
        # 兜底：用乘除法实现向零截断
        sign = 1 if x >= 0 else -1
        x_abs = abs(x)
        return sign * int(x_abs * 10000) / 10000.0

def parse_longest_from_dimension(dim_str: str) -> Optional[float]:
    """
    从形如 "L*W*H" 的字符串里解析数值，返回最大边（float）。
    - 允许空格，如 "12 * 7.5 * 3"
    - 如果 token 含单位（cm/m/in 等），仍尝试抽出第一个浮点数
    - 不做单位换算：视作"模型原始单位"
    """
    if not dim_str or not isinstance(dim_str, str):
        return None
    parts = re.split(r"\s*\*\s*", dim_str.strip())
    vals = []
    for p in parts:
        # 抽取 token 中的第一个浮点数
        m = re.search(r"[-+]?\d+(?:\.\d+)?", p)
        if m:
            try:
                vals.append(float(m.group(0)))
            except Exception:
                pass
    return max(vals) if vals else None

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def compute_scales(
    longest_path: Path,
    dim_path: Path,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    根据两个输入文件计算 scale:
    scale = longest_m / dimension_max
    返回 (scales字典, 错误信息字典)
    """
    longest_data = load_json(longest_path)
    dim_data = load_json(dim_path)

    # 期望结构：
    # longest_data: { uuid: {"longest_m": float}, ... }
    # dim_data:     { uuid: {"dimension": "L*W*H", ...}, ... }
    scales: Dict[str, float] = {}
    errors: Dict[str, str] = {}

    for uuid, rec in longest_data.items():
        try:
            if not isinstance(rec, dict) or "longest_m" not in rec:
                errors[uuid] = "missing longest_m"
                continue
            longest_m = float(rec["longest_m"])

            src = dim_data.get(uuid)
            if not isinstance(src, dict) or "dimension" not in src:
                errors[uuid] = "missing dimension in filtered file"
                continue

            dim_max = parse_longest_from_dimension(src["dimension"])
            if dim_max is None or dim_max <= 0:
                errors[uuid] = f"invalid dimension: {src.get('dimension')!r}"
                continue

            scale_raw = longest_m / float(dim_max)
            scale = truncate_to_4(scale_raw)  # 👈 向零截断到最多 4 位小数
            scales[uuid] = scale
        except Exception as e:
            errors[uuid] = f"exception: {e!r}"

    return scales, errors

def main():
    ap = argparse.ArgumentParser(description="Compute scale from longest_m and dimension.")
    ap.add_argument(
        "-l", "--longest",
        type=str,
        default="robotwin_longest_m_by_gpt41.json",
        help="path to JSON produced by the model (uuid -> {longest_m})"
    )
    ap.add_argument(
        "-d", "--dims",
        type=str,
        default="filtered_robotwin_dim_img.json",
        help="path to source JSON containing dimension strings (uuid -> {dimension})"
    )
    ap.add_argument(
        "-o", "--output",
        type=str,
        default="robotwin_scale_from_longest.json",
        help="output JSON path for scales (uuid -> scale)"
    )
    ap.add_argument(
        "--errors",
        type=str,
        default=None,
        help="optional path to save an errors JSON (uuid -> reason). If omitted, will write alongside output with suffix '_errors.json' when there are errors."
    )
    args = ap.parse_args()

    longest_path = Path(args.longest)
    dim_path = Path(args.dims)
    out_path = Path(args.output)

    scales, errors = compute_scales(longest_path, dim_path)

    # 保存结果
    out_path.write_text(json.dumps(scales, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {len(scales)} scales -> {out_path}")

    if errors:
        err_path = Path(args.errors) if args.errors else out_path.with_name(out_path.stem + "_errors.json")
        err_path.write_text(json.dumps({"count": len(errors), "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[WARN] {len(errors)} issues -> {err_path}")

if __name__ == "__main__":
    main()
