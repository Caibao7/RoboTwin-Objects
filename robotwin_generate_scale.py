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
MAX_WORKERS   = int(os.environ.get("WORKERS", "6"))
MAX_RETRIES   = 3

# grok prompt: 
SYSTEM_PROMPT = r"""
You are a 3D asset scale estimator.

Inputs per object:
- One IMAGE of the object (rendered or photo).
- dimension: L*W*H (unit unknown)

Output: return a strict JSON with exactly this field:
- scale (float; real dimensions in m = input dimension * scale)

Grounding:
- Base on the physical 3D object in the real world to estimate the typical scale.
- Estimate typical real longest side (m) from the recognized object.
- Use longest-side alignment: scale = typical_longest_m / model_longest_in_input_units.
- Aim for conservative estimates to avoid over-scaling; use common real-world sizes for everyday objects (e.g., chair ~0.5-1m height, table ~0.7-1m height, car ~4-5m length).
- If uncertain, make an estimate based on the closest common object category.
- Never output null, NaN, or non-numeric values.
- Ensure the resulting real dimensions (dimension * scale) are plausible and not excessively large.

Formatting rules:
- Output JSON only; no extra text.
- 'scale' is a single scalar float (no units).

Example output:
{
  "scale": 0.05
}
"""


client = OpenAI()  # 依赖 OPENAI_API_KEY


def is_responses_model(model: str) -> bool:
    m = (model or "").lower()
    # 所有 o4 家族与 gpt-5 家族都当作 Responses API 模式
    return m.startswith(("o4", "gpt-5"))


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
        "Return JSON only."
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
def postprocess(record: dict) -> dict:
    # 轻度 clamp，防止极端值
    def clamp_num(x, lo, hi, default):
        try:
            x = float(x)
            return max(lo, min(hi, x))
        except:
            return default

    if not isinstance(record, dict):
        return record
    if "scale" in record:
        record["scale"] = clamp_num(record["scale"], 1e-8, 1e3, 1.0)
        
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

    bar = tqdm(total=len(items), desc="GPT-5 annotating", unit="obj") if tqdm else None

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
    ap.add_argument("-o", "--output", type=str, default="robotwin_scale_generated_by_gpt41.json", help="output JSON file")
    ap.add_argument("-m", "--model",  type=str, default=DEFAULT_MODEL, help="model name, e.g., gpt-5 / gpt-5-mini / gpt-4o")
    ap.add_argument("-w", "--workers", type=int, default=MAX_WORKERS, help="max concurrent workers")
    ap.add_argument("-n", "--limit",  type=int, default=None, help="only process first N objects")
    ap.add_argument("--dry-run", action="store_true", help="run first 3 items synchronously and print")
    args = ap.parse_args()

    run_batch(Path(args.input), Path(args.output), model=args.model, limit=args.limit, workers=args.workers, dry_run=args.dry_run)