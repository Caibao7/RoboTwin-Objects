# -*- coding: utf-8 -*-
import os, json, re, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple
from openai import OpenAI

# 进度条（可选）
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ================== Prompt ==================
SYSTEM_PROMPT = r"""
You are an expert annotator for scene entities. Your task is to classify each object into exactly one of:
- "DynamicEntities": objects that are commonly **moved, manipulated, picked up, or repositioned** during normal use.
- "StructuralEntities": objects that are generally **not moved** in day-to-day use (fixed, built-in, heavy, large, or typically left in place).

You will receive, per object:
- object_name (string)
- category (string)
- real_size: real world bounding box. array [x, y, z] in meters (AABB; x=width, y=depth, z=height)
- density (float, unit is g/cm^3)
- static_friction (float)
- dynamic_friction (float)
- restitution (float)
- Basic_description (one sentence)
- Functional_description (list of short phrases)

Decision guidelines (signals, not hard rules):
- Building structure / fixed installation / large heavy furniture or appliances, prefer **StructuralEntities**.
- Handheld / manipulable / portable / frequently repositioned objects, prefer **DynamicEntities**.

Output format:
- Return **JSON only** with exactly this single field:
  { "tags": "DynamicEntities" } or { "tags": "StructuralEntities" }
- No extra text, no explanations.
"""

# ================== 配置 ==================
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")
MAX_WORKERS   = int(os.environ.get("WORKERS", "2"))
MAX_RETRIES   = 3

client = OpenAI()

# ------------------ LLM 调用 ------------------
def _token_kw(model: str, n: int):
    m = (model or "").lower()
    if m.startswith("gpt-5"):
        return {"max_completion_tokens": n}
    return {"max_tokens": n}

def _temp_kw(model: str, t: float):
    m = (model or "").lower()
    if m.startswith("gpt-5"):
        return {}
    return {"temperature": t}

def ask_llm_for_tag(payload: Dict[str, Any], model: str, timeout: int = 60) -> str:
    user_text = json.dumps(payload, ensure_ascii=False)
    last_err = "unknown"
    for _ in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_text},
                ],
                response_format={"type": "json_object"},
                **_temp_kw(model, 0.0),
                **_token_kw(model, 64),
                timeout=timeout,
            )
            txt = (resp.choices[0].message.content or "").strip()
            obj = json.loads(txt)
            tag = (obj.get("tags") or "").strip()
            if tag in ("DynamicEntities", "StructuralEntities"):
                return tag
            m = re.search(r'"tags"\s*:\s*"(?P<t>DynamicEntities|StructuralEntities)"', txt)
            if m:
                return m.group("t")
            last_err = f"unexpected content: {txt[:200]}"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(f"LLM failed to return valid tag. Last error: {last_err}")

# ------------------ 数据准备 ------------------
def _to_float(x, default=None):
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else default

def build_payload_from_info(info: Dict[str, Any]) -> Dict[str, Any]:
    rs = info.get("real_size")
    if isinstance(rs, (list, tuple)) and len(rs) == 3:
        try:
            real_size = [float(rs[0]), float(rs[1]), float(rs[2])]
        except Exception:
            real_size = None
    else:
        real_size = None

    return {
        "object_name": info.get("object_name"),
        "category": info.get("category"),
        "real_size": real_size,
        "density": _to_float(info.get("density")),
        "static_friction": _to_float(info.get("static_friction")),
        "dynamic_friction": _to_float(info.get("dynamic_friction")),
        "restitution": _to_float(info.get("restitution")),
        "Basic_description": info.get("Basic_description"),
        "Functional_description": info.get("Functional_description"),
    }

# ------------------ 主流程 ------------------
def process_one(uuid: str, meta: Dict[str, Any], info_map: Dict[str, Any], model: str) -> Tuple[str, str, str, bool]:
    old_tag = (meta.get("tags") or ["DynamicEntities"])[0]
    info = info_map.get(uuid)
    if not isinstance(info, dict):
        return uuid, old_tag, old_tag, False
    try:
        payload = build_payload_from_info(info)
        new_tag = ask_llm_for_tag(payload, model=model)
        return uuid, old_tag, new_tag, True
    except Exception:
        return uuid, old_tag, old_tag, True

def run(objects_json: Path, info_json: Path, output_json: Path, model: str, limit: int = None, workers: int = MAX_WORKERS):
    objects_data = json.loads(objects_json.read_text(encoding="utf-8"))
    info_map     = json.loads(info_json.read_text(encoding="utf-8"))

    items = list(objects_data.items())
    if limit is not None and limit > 0:
        items = items[:limit]

    updates: Dict[str, Tuple[str, str, bool]] = {}

    # --- 进度条开始 ---
    bar = tqdm(total=len(items), desc="LLM tagging", unit="obj") if tqdm else None

    same_count = 0
    diff_count = 0
    info_missing = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for uuid, meta in items:
            futs.append(ex.submit(process_one, uuid, meta, info_map, model))
        for fut in as_completed(futs):
            u, old_tag, new_tag, has_info = fut.result()
            updates[u] = (old_tag, new_tag, has_info)

            # 实时统计（可选）
            if not has_info:
                info_missing += 1
                same_count += 1  # 缺信息时不改
            else:
                if new_tag and new_tag in ("DynamicEntities", "StructuralEntities"):
                    if new_tag != old_tag:
                        diff_count += 1
                    else:
                        same_count += 1
                else:
                    same_count += 1

            if bar:
                # 每完成一个对象就前进一步
                bar.update(1)
                # 可选：显示简短统计
                if (same_count + diff_count) % 25 == 0:
                    bar.set_postfix(same=same_count, diff=diff_count, miss=info_missing)

    if bar:
        bar.close()
    # --- 进度条结束 ---

    # 应用更新
    for uuid, (old_tag, new_tag, has_info) in updates.items():
        if has_info and new_tag and new_tag in ("DynamicEntities", "StructuralEntities") and new_tag != old_tag:
            objects_data[uuid]["tags"] = [new_tag]

    output_json.write_text(json.dumps(objects_data, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(items)
    print(f"Saved to {output_json}")
    print(f"Total objects processed: {total}")
    print(f"Tags SAME:   {same_count}")
    print(f"Tags DIFFER: {diff_count}")
    print(f"Info missing: {info_missing}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=str, default="robotwin_objects.json")
    parser.add_argument("--info", type=str, default="robotwin_info_generated_by_llm.json",
        help="path to robotwin_info_generated_by_llm.json")
    parser.add_argument("--output", type=str, default="robotwin_objects_new.json")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("-n", "--limit", type=int, default=None)
    parser.add_argument("-w", "--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    run(Path(args.objects), Path(args.info), Path(args.output),
        model=args.model, limit=args.limit, workers=args.workers)

if __name__ == "__main__":
    main()
