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
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")   # 可改 gpt-5-mini / gpt-4.1 / gpt-4o / gpt-4o-mini
MAX_WORKERS   = int(os.environ.get("WORKERS", "2"))
MAX_RETRIES   = 3

# ----------------------------- 新系统提示词 -----------------------------
# 任务：估计“实物三维包围盒的最长边（单位：米）”，仅输出 JSON: {"longest_m": <float>}
SYSTEM_PROMPT = r"""
You are a 3D asset longest-side estimator.

Inputs per object:
- One IMAGE of the object (rendered or photo).

Your single task:
- Predict the real-world longest side length of the depicted object's 3D bounding box, in meters.

Output:
Return a strict JSON with exactly this field:
- longest_m (float; meters)

Grounding & rules:
- Base your estimate on the physical real-world object the image depicts.
- Estimate using **typical, conservative real-world sizes**.
- If the picture looks like a toy/miniature, **still** estimate as the corresponding real object.
- When uncertain, choose the **lower end** of the common size range (around 10–20% below a rough midpoint), not the upper end.
- Avoid overestimation from close-up shots: an object filling the frame is **not** necessarily large.
- Never output null, NaN, or non-numeric values.
- Ensure the result is plausible (not excessively large or tiny for the recognized object).

Formatting:
- Output JSON only; no extra text.
- Exactly one key: "longest_m" (float).

Example output:
{
  "longest_m": 1.2
}
"""

client = OpenAI()  # 依赖 OPENAI_API_KEY

# ----------------------------- 工具函数 -----------------------------
def is_gpt5(model: str) -> bool:
    return (model or "").lower().startswith("gpt-5")


def out_tok_kw(model: str, n: int):
    """
    对 gpt-5（Responses API）使用 max_output_tokens；保证至少 16。
    其它模型不走这里（见 token_kw）。
    """
    if is_gpt5(model):
        return {"max_output_tokens": max(n, 16)}
    return token_kw(model, n)


def temp_kw(model: str, t: float):
    m = (model or "").lower()
    # gpt-5 系列不传或固定 1
    if m.startswith("gpt-5"):
        return {}  # 或者 return {"temperature": 1}
    return {"temperature": t}


def token_kw(model: str, n: int):
    """
    为不同模型返回正确的 token 上限参数：
    - GPT-5 系列（Responses API）在 out_tok_kw 中统一处理
    - 其它（如 gpt-4.1 / gpt-4o / gpt-4o-mini）：使用 max_tokens
    """
    m = (model or "").lower()
    if m.startswith("gpt-5"):
        # 不在 chat.completions 使用
        return {"max_completion_tokens": n}
    return {"max_tokens": n}


def extract_text_from_response(resp) -> str:
    """
    兼容多版本/多模型的 Responses API 返回。
    优先 output_text；再扫 output[*].content[*].text(.value)；再兜底 message/choices。
    """
    # 1) 官方聚合文本
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    # 2) 明细结构（Responses API）
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

    # 3) 极端兜底
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


def parse_longest(dim_str: str):
    if not dim_str:
        return None
    parts = re.split(r"\s*\*\s*", dim_str.strip())
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except:
            pass
    return max(vals) if vals else None


def strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if s.startswith("```") and s.endswith("```"):
        s = s.split("\n", 1)[-1]
        s = s.rsplit("\n", 1)[0]
    return s.strip()


def extract_json_block(s: str) -> str:
    """
    从任意文本中提取首个平衡的 {...} JSON 块。
    不使用递归正则，兼容 Python 标准库。
    """
    if not s:
        return ""
    s = strip_code_fences(s)
    start = s.find("{")
    if start == -1:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return ""


def encode_image_to_data_url(path: str) -> str:
    """
    将本地图片转为 data URL，便于直接走多模态输入。
    支持常见格式（png/jpg/webp等）。
    """
    if not path:
        return ""
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_user_message(dim: str, image_ref: str) -> List[Dict[str, Any]]:
    """
    仅使用 dimension 与图片构造多模态消息（Responses API 风格）
    """
    longest = parse_longest(dim) if dim else None
    longest_txt = f"(model_longest={longest})" if longest is not None else "(model_longest=unknown)"

    text = (
        f'dimension: "{dim or ""}" {longest_txt}\n\n'
        "Goal: predict the real-world longest bounding-box side length of the depicted object, in meters (m).\n"
        "Do NOT compute or output any 'scale'. Return JSON only."
    )

    # 处理图片
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
    """
    将 Responses API 风格的多模态 messages 转为 Chat Completions 可接受的格式：
    - input_text -> {"type": "text", "text": "..."}
    - input_image -> {"type": "image_url", "image_url": {"url": "..."}}
    """
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
    """
    对 gpt-5 走 Responses API（支持多模态 input），对其它模型走 chat.completions（多模态转换）。
    返回纯文本（可能是 JSON 字符串）。
    """
    if is_gpt5(model):
        resp = client.responses.create(
            model=model,
            input=messages,             # Responses API 风格
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


def call_gpt(dim: str, image_ref: str, model: str, timeout: int = 120) -> dict:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            messages = build_user_message(dim, image_ref)
            text = request_text(
                model, messages,
                json_mode=not is_gpt5(model),
                timeout=timeout,
                max_tokens=100,  # 简化输出，减少tokens
            )
            return json.loads(text)
        except Exception as e_first:
            last_err = e_first
            time.sleep((2 ** (attempt - 1)) + random.random())
            try:
                messages = build_user_message(dim, image_ref)
                text = request_text(
                    model, messages,
                    json_mode=False,
                    timeout=timeout,
                    max_tokens=100,
                )
                # JSON 兜底：截取首个完整 {...} 块
                try:
                    return json.loads(text)
                except Exception:
                    block = extract_json_block(text)
                    if block:
                        return json.loads(block)
                    raise
            except Exception as e_second:
                last_err = e_second
                time.sleep((2 ** (attempt - 1)) + random.random())
    raise RuntimeError(f"All attempts failed. Last error: {last_err}")


# ----------------------------- 后处理与探针 -----------------------------
def postprocess(record: dict) -> dict:
    # 轻度 clamp，防止极端值；对常见物体，0.01m~100m 足够宽
    def clamp_num(x, lo, hi, default):
        try:
            x = float(x)
            return max(lo, min(hi, x))
        except:
            return default

    if not isinstance(record, dict):
        return record
    if "longest_m" in record:
        record["longest_m"] = clamp_num(record["longest_m"], 0.01, 100.0, 1.0)
    return record


def probe_model(model: str):
    """
    极简自检：要求输出 {"ok": true}
    - gpt-5：Responses API，且不传 temperature/response_format
    - 其它模型：chat.completions + JSON 模式
    """
    # 构造与正式流程一致的 system/user 结构（无图片）
    sys_msg = {"role": "system", "content": [{"type": "input_text", "text": "Return valid JSON only."}]}
    user_msg = {"role": "user",   "content": [{"type": "input_text", "text": 'Reply with {"ok": true} exactly.'}]}
    messages = [sys_msg, user_msg]

    try:
        text = request_text(
            model, messages,
            json_mode=not is_gpt5(model),
            timeout=60,
            max_tokens=32,   # >=16 for gpt-5
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

    # 模型可用性自检
    ok, err = probe_model(model)
    if not ok:
        print(f"[Model probe] model={model} failed: {err}")
        print("Tip: try --model gpt-5-mini / gpt-4o / gpt-4.1 / gpt-4o-mini ；或升级 SDK。")
        return

    results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    # dry-run 打印前 3 条，方便快速验证
    if dry_run:
        print("[Dry-run] Running the first 3 objects synchronously...")
        for obj_id, obj in items[:3]:
            dim   = obj.get("dimension", "")
            img   = obj.get("image", "")
            try:
                data = call_gpt(dim, img, model=model)
                data = postprocess(data)
                print(obj_id, "=>", json.dumps(data, ensure_ascii=False))
            except Exception as e:
                print(obj_id, "ERROR:", repr(e))
        return

    bar = tqdm(total=len(items), desc="GPT annotating longest_m", unit="obj") if tqdm else None

    def work(kv: Tuple[str, Dict[str, Any]]):
        obj_id, obj = kv
        dim = obj.get("dimension", "")
        img = obj.get("image", "")
        data = call_gpt(dim, img, model=model)
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

    # 保存
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} items to {output_path}")
    if errors:
        err_p = output_path.with_name(output_path.stem + "_errors.json")
        err_p.write_text(json.dumps({"count": len(errors), "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(errors)} errors to {err_p}")


# ----------------------------- 入口 -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input",  type=str, default="filtered_robotwin_dim_img.json", help="input JSON file")
    ap.add_argument("-o", "--output", type=str, default="robotwin_longest_m_by_model.json", help="output JSON file")
    ap.add_argument("-m", "--model",  type=str, default=DEFAULT_MODEL, help="model name, e.g., gpt-5 / gpt-5-mini / gpt-4o")
    ap.add_argument("-w", "--workers", type=int, default=MAX_WORKERS, help="max concurrent workers")
    ap.add_argument("-n", "--limit",  type=int, default=None, help="only process first N objects")
    ap.add_argument("--dry-run", action="store_true", help="run first 3 items synchronously and print")
    args = ap.parse_args()

    run_batch(Path(args.input), Path(args.output), model=args.model, limit=args.limit, workers=args.workers, dry_run=args.dry_run)

