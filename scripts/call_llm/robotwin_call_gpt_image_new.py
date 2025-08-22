# -*- coding: utf-8 -*-
import os, json, re, time, random, argparse, base64, mimetypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple, List

from openai import OpenAI
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ----------------------------- 配置 -----------------------------
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")  # 可改 gpt-5-mini / gpt-4.1 / gpt-4o / gpt-4o-mini
MAX_WORKERS   = int(os.environ.get("WORKERS", "1"))
MAX_RETRIES   = 3

SYSTEM_PROMPT = r"""
You are a 3D asset semantics & physical-attributes annotator.

Inputs per object:
- One IMAGE: a single composite image that shows the same object from six viewpoints (front/back/left/right/top/bottom).
- A string robotwin_name like "039_mug_2" or "081_playingcards_2".

Task:
Infer the object semantics and basic physical attributes by **combining visual evidence from the six-view image** and **the hint from robotwin_name** (the base noun in robotwin_name is highly indicative, e.g., "mug", "playingcards", "french-fries").

Output: return a strict JSON with **exactly** the following fields (no extras):
- object_name (1–4 words, natural but joined by underscores, all lowercase; refine robotwin_name into a human-friendly name, e.g., "ceramic_mug", "deck_of_cards")
- category (one simple word, lowercase; e.g., "mug", "cards", "box")
- real_size (array of 3 floats [x, y, z] in meters; the object's axis-aligned bounding box in the real world: x=width (left-right), y=depth (front-back), z=height (bottom-top))
- density (float, g/cm^3; dominant material estimate)
- static_friction (float, coefficient μ_s on a dry clean WOOD tabletop)
- dynamic_friction (float, coefficient μ_k on a dry clean WOOD tabletop; usually ~0.75 of μ_s for similar contact)
- restitution (float, 0–1; effective normal COR when dropped on wood from small height)
- Basic_description (one concise sentence)
- Functional_description (a **list of concise functions**, sorted from most to least likely for this exact object; each item is a short phrase like "holds hot beverages")

Grounding & rules:
- Use both robotwin_name and image cues; when they disagree, favor the **image** but keep robotwin_name as a strong prior.
- **real_size estimation policy:**
  - Output [x, y, z] in **meters** for the axis-aligned 3D bounding box, with x=width (left-right), y=depth (front-back), z=height (bottom-top).
  - Base the estimate on **typical, real-world sizes** for the recognized category.
  - Use the six views to infer proportions.
- Keep numbers as plain floats (no units in strings). Output JSON only.
- One sentence for each description; be specific to the object seen.

Reference ranges (pick a reasonable single value, not a range):
- Common densities (g/cm^3): paper/cardboard ~0.60; plastics: HDPE 0.94, ABS 1.06, Nylon 1.14, PET 1.38, PVC 1.40; wood ~0.65 (typ.); soda-lime glass 2.50; ceramic 2.5–3.9 (use ~2.8 if unsure); steel 7.8; rubber (natural) 0.92; silicone 1.2.
- Static friction μ_s on WOOD (dry/clean): hard plastic ~0.40; soft plastic ~0.55; rubber ~0.70–0.95; paper/cardboard ~0.45; ceramic/glass on wood ~0.40; metal on wood ~0.35–0.50.
- Dynamic friction μ_k ≈ 0.7–0.85 * μ_s (choose a consistent reasonable value).
- Restitution (on wood): ceramic/glass 0.5–0.7; hard plastic 0.4–0.6; rubber 0.7–0.9; paper/cardboard 0.2–0.4; metal 0.4–0.6.

Few-shot exemplars (for style/scale only; DO NOT copy blindly—use the **current** image + name):
Example (robotwin_name="039_mug_2"):
{
  "object_name": "ceramic_mug",
  "category": "mug",
  "real_size": [0.085, 0.085, 0.11],
  "density": 2.8,
  "static_friction": 0.40,
  "dynamic_friction": 0.30,
  "restitution": 0.55,
  "Basic_description": "A cylindrical mug with a handle and smooth surface.",
  "Functional_description": [
    "drink container",
    "food holder",
    "heating vessel",
    "storage container",
    "measuring tool",
    "decorative item",
    "improvised use"
  ]
}
"""

client = OpenAI()  # 依赖 OPENAI_API_KEY

# ----------------------------- 工具函数 -----------------------------
def is_gpt5(model: str) -> bool:
    return (model or "").lower().startswith("gpt-5")

def is_responses_model(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith(("o4", "gpt-5"))

def out_tok_kw(model: str, n: int):
    if is_gpt5(model):
        return {"max_output_tokens": max(n, 16)}
    return token_kw(model, n)

def temp_kw(model: str, t: float):
    m = (model or "").lower()
    if m.startswith("gpt-5"):
        return {}
    return {"temperature": t}

def token_kw(model: str, n: int):
    m = (model or "").lower()
    if m.startswith("gpt-5"):
        return {"max_completion_tokens": n}
    return {"max_tokens": n}

def extract_text_from_response(resp) -> str:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    try:
        out = getattr(resp, "output", None) or []
        parts = []
        for item in out:
            content = getattr(item, "content", None) or []
            for c in content:
                t = getattr(c, "text", None)
                if isinstance(t, str) and t.strip():
                    parts.append(t)
                else:
                    v = getattr(t, "value", None) if t is not None else None
                    if isinstance(v, str) and v.strip():
                        parts.append(v)
        if parts:
            return "".join(parts).strip()
    except Exception:
        pass
    try:
        msg = getattr(resp, "message", None)
        if msg:
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                s = "".join([getattr(x, "text", "") or getattr(getattr(x, "text", None), "value", "") or "" for x in content])
                if s.strip():
                    return s.strip()
    except Exception:
        pass
    return ""

def encode_image_to_data_url(path: str) -> str:
    if not path:
        return ""
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def build_user_message(robotwin_name: str, image_ref: str) -> List[Dict[str, Any]]:
    """
    使用 robotwin_name 与六视角拼接图片构造多模态消息（Responses API 风格）
    """
    text = (
        f'robotwin_name: "{robotwin_name or ""}"\n'
        "The image is a 6-view composite of the same object (front/back/left/right/top/bottom).\n"
        "Return JSON only."
    )

    image_url = ""
    if image_ref:
        if image_ref.startswith(("http://", "https://", "data:")):
            image_url = image_ref
        else:
            try:
                image_url = encode_image_to_data_url(image_ref)
            except Exception:
                image_url = ""

    user_content = [{"type": "input_text", "text": text}]
    if image_url:
        user_content.append({"type": "input_image", "image_url": image_url})

    return [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": user_content},
    ]

def to_chat_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", [])
        new_content = []
        if isinstance(content, list):
            for c in content:
                t = c.get("type")
                if t == "input_text":
                    new_content.append({"type": "text", "text": c.get("text", "")})
                elif t == "input_image":
                    url = c.get("image_url", "")
                    if url:
                        new_content.append({"type": "image_url", "image_url": {"url": url}})
        elif isinstance(content, str):
            new_content.append({"type": "text", "text": content})
        else:
            new_content.append({"type": "text", "text": ""})
        out.append({"role": role, "content": new_content})
    return out

# ----------------------------- 核心调用 -----------------------------
def request_text(model: str, messages, *, json_mode: bool, timeout: int, max_tokens: int) -> str:
    if is_gpt5(model):
        resp = client.responses.create(
            model=model,
            input=messages,
            **out_tok_kw(model, max_tokens),
            timeout=timeout,
        )
        return extract_text_from_response(resp) or ""
    else:
        chat_messages = to_chat_messages(messages)
        resp = client.chat.completions.create(
            model=model,
            messages=chat_messages,
            **({"response_format": {"type": "json_object"}} if json_mode else {}),
            **temp_kw(model, 0.2),
            **token_kw(model, max_tokens),
            timeout=timeout,
        )
        return (resp.choices[0].message.content or "").strip()

def call_gpt(robotwin_name: str, image_ref: str, model: str, timeout: int = 120) -> dict:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            messages = build_user_message(robotwin_name, image_ref)
            text = request_text(
                model, messages,
                json_mode=not is_gpt5(model),
                timeout=timeout,
                max_tokens=600,
            )
            return json.loads(text)
        except Exception as e_first:
            last_err = e_first
            time.sleep((2 ** (attempt - 1)) + random.random())
            try:
                messages = build_user_message(robotwin_name, image_ref)
                text = request_text(
                    model, messages,
                    json_mode=False,
                    timeout=timeout,
                    max_tokens=600,
                )
                try:
                    return json.loads(text)
                except Exception:
                    m = re.search(r"\{(?:[^{}]|(?R))*\}", text, flags=re.S)
                    if m:
                        return json.loads(m.group(0))
                    raise
            except Exception as e_second:
                last_err = e_second
                time.sleep((2 ** (attempt - 1)) + random.random())
    raise RuntimeError(f"All attempts failed. Last error: {last_err}")

# ----------------------------- 后处理与探针 -----------------------------
def _clamp_float(x, lo, hi, default):
    try:
        v = float(x)
        return max(lo, min(hi, v))
    except:
        return default

def _coerce_real_size(val):
    """
    期望 [x, y, z] 米；限制到 [0.001, 10] 之间；长度为 3。
    """
    if isinstance(val, (list, tuple)) and len(val) == 3:
        return [
            _clamp_float(val[0], 1e-3, 10.0, 0.1),
            _clamp_float(val[1], 1e-3, 10.0, 0.1),
            _clamp_float(val[2], 1e-3, 10.0, 0.1),
        ]
    # 尝试从字符串解析
    if isinstance(val, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", val)
        if len(nums) >= 3:
            return [
                _clamp_float(nums[0], 1e-3, 10.0, 0.1),
                _clamp_float(nums[1], 1e-3, 10.0, 0.1),
                _clamp_float(nums[2], 1e-3, 10.0, 0.1),
            ]
    return [0.1, 0.1, 0.1]

def _clamp_float(x, lo, hi, default):
    try:
        v = float(x)
        return max(lo, min(hi, v))
    except:
        return default

def _coerce_real_size(val):
    # 期望 [x, y, z] 米；限制到 [0.001, 10] 之间；长度为 3。
    if isinstance(val, (list, tuple)) and len(val) == 3:
        return [
            _clamp_float(val[0], 1e-3, 10.0, 0.1),
            _clamp_float(val[1], 1e-3, 10.0, 0.1),
            _clamp_float(val[2], 1e-3, 10.0, 0.1),
        ]
    if isinstance(val, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", val)
        if len(nums) >= 3:
            return [
                _clamp_float(nums[0], 1e-3, 10.0, 0.1),
                _clamp_float(nums[1], 1e-3, 10.0, 0.1),
                _clamp_float(nums[2], 1e-3, 10.0, 0.1),
            ]
    return [0.1, 0.1, 0.1]

def _to_function_list(val, max_len=6):
    """把 Functional_description 规整为按可能性排序的列表"""
    if isinstance(val, list):
        items = [str(x).strip() for x in val if str(x).strip()]
    else:
        s = str(val or "")
        # 允许用逗号/分号/顿号/斜杠/换行/句号分割
        items = [x.strip(" -•·") for x in re.split(r"[;,/|｜、\n]+|[。.!?]", s) if x.strip()]
    # 去重（不区分大小写）并截断
    seen, out = set(), []
    for it in items:
        it = re.sub(r"\s+", " ", it)
        key = it.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it)
        if len(out) >= max_len:
            break
    return out


def postprocess(record: dict) -> dict:
    if not isinstance(record, dict):
        return record

    # 需要的字段（已去掉 Movement_description / Placement_description）
    want_keys = {
        "object_name", "category", "real_size", "density",
        "static_friction", "dynamic_friction", "restitution",
        "Basic_description", "Functional_description"
    }
    for k in want_keys:
        record.setdefault(k, "" if "description" in k.lower() else None)

    # 数值规整
    record["real_size"] = _coerce_real_size(record.get("real_size"))
    record["density"] = _clamp_float(record.get("density"), 0.05, 25.0, 1.0)
    record["static_friction"]  = _clamp_float(record.get("static_friction"), 0.0, 2.0, 0.5)
    record["dynamic_friction"] = _clamp_float(record.get("dynamic_friction"), 0.0, 2.0, 0.4)
    record["restitution"]      = _clamp_float(record.get("restitution"), 0.0, 1.0, 0.5)

    # 文本规整
    if isinstance(record.get("object_name"), str):
        record["object_name"] = record["object_name"].strip().lower().replace(" ", "_")
        record["object_name"] = re.sub(r"__+", "_", record["object_name"]).strip("_") or "object"
    if isinstance(record.get("category"), str):
        record["category"] = record["category"].strip().lower()

    # 描述：Basic 保留一句；Functional 转为 list
    v = str(record.get("Basic_description", "") or "").strip()
    first_sentence = re.split(r"[。.!?]\s*", v)
    record["Basic_description"] = (first_sentence[0] if first_sentence and first_sentence[0] else v)[:300]

    funcs = _to_function_list(record.get("Functional_description"))
    record["Functional_description"] = funcs  # 允许空列表

    return record


def probe_model(model: str):
    sys_msg = {"role": "system", "content": [{"type": "input_text", "text": "Return valid JSON only."}]}
    user_msg = {"role": "user",   "content": [{"type": "input_text", "text": 'Reply with {"ok": true} exactly.'}]}
    messages = [sys_msg, user_msg]
    try:
        text = request_text(
            model, messages,
            json_mode=not is_gpt5(model),
            timeout=60,
            max_tokens=32,
        )
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and obj.get("ok") is True:
                return True, None
        except Exception:
            if '"ok":true' in (text or "").replace(" ", "").lower():
                return True, None
        return False, f"Non-JSON or unexpected content: {repr(text)[:200]}"
    except Exception as e:
        msg = str(e)
        if "Use 'max_completion_tokens' instead" in msg and "max_tokens" in msg:
            return False, msg
        return False, msg

# ----------------------------- 批处理主流程 -----------------------------
def run_batch(input_path: Path, output_path: Path, model: str, limit: int = None, workers: int = MAX_WORKERS, dry_run: bool=False):
    src = json.loads(input_path.read_text(encoding="utf-8"))
    items = sorted(src.items(), key=lambda kv: kv[0])
    if limit is not None and limit > 0:
        items = items[:limit]

    ok, err = probe_model(model)
    if not ok:
        print(f"[Model probe] model={model} failed: {err}")
        print("Tip: try --model gpt-5-mini / gpt-4o / gpt-4.1 / gpt-4o-mini ；或升级 SDK。")
        return

    results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    if dry_run:
        print("[Dry-run] Running the first 3 objects synchronously...")
        for obj_id, obj in items[:3]:
            name = obj.get("robotwin_name", "")
            img  = obj.get("image", "")
            try:
                data = call_gpt(name, img, model=model)
                data = postprocess(data)
                print(obj_id, "=>", json.dumps(data, ensure_ascii=False))
            except Exception as e:
                print(obj_id, "ERROR:", repr(e))
        return

    bar = tqdm(total=len(items), desc="GPT annotating", unit="obj") if tqdm else None

    def work(kv: Tuple[str, Dict[str, Any]]):
        obj_id, obj = kv
        name = obj.get("robotwin_name", "")
        img  = obj.get("image", "")
        data = call_gpt(name, img, model=model)
        data = postprocess(data)
        return obj_id, data

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut2id = {}
        for kv in items:
            fut = ex.submit(work, kv)
            fut2id[fut] = kv[0]
        for fut in as_completed(fut2id):
            obj_id = fut2id[fut]
            try:
                k, data = fut.result()
                results[k] = data
            except Exception as e:
                errors[obj_id] = repr(e)
            if bar:
                bar.update(1)
                if errors and len(errors) % 50 == 0:
                    bar.set_postfix(err=len(errors))
    if bar:
        bar.close()

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} items to {output_path}")
    if errors:
        err_p = output_path.with_name(output_path.stem + "_errors.json")
        err_p.write_text(json.dumps({"count": len(errors), "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(errors)} errors to {err_p}")

# ----------------------------- 入口 -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input",  type=str, default="filtered_robotwin_img_with_name.json", help="input JSON file")
    ap.add_argument("-o", "--output", type=str, default="robotwin_info_generated_by_llm.json", help="output JSON file")
    ap.add_argument("-m", "--model",  type=str, default=DEFAULT_MODEL, help="model name, e.g., gpt-5-mini / gpt-4o / gpt-4.1")
    ap.add_argument("-w", "--workers", type=int, default=MAX_WORKERS, help="max concurrent workers")
    ap.add_argument("-n", "--limit",  type=int, default=None, help="only process first N objects")
    ap.add_argument("--dry-run", action="store_true", help="run first 3 items synchronously and print")
    args = ap.parse_args()

    run_batch(Path(args.input), Path(args.output), model=args.model, limit=args.limit, workers=args.workers, dry_run=args.dry_run)
