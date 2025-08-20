#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to validate real sizes of RobotWin objects using LLM.
Reads robotwin_real_sizes.json and asks LLM to judge if the real_size is proper 
compared to the actual size of objects in the real world.

Usage:
    python validate_real_sizes.py --input robotwin_real_sizes.json --output size_validation_results.json
    python validate_real_sizes.py --dry-run --limit 5
"""

import os, json, re, time, random, argparse
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

# Size validation prompt
SYSTEM_PROMPT = r"""
You are a real-world object size validator.

Your task is to judge whether the calculated real_size of an object is reasonable compared to typical real-world dimensions of that object.

Input per object:
- object_name: name of the object
- category: object category
- dimension: string, "x*y*z" (model units)
- scale: a list of values (applied per axis)
- real_size: calculated dimensions in format "length*width*height" (unit: meters)

Output: return a strict JSON with exactly these fields:
- is_proper (boolean): true if the size is reasonable, false if not
- assessment (string): if is_proper is false, specify "too_big" or "too_small"
- typical_size_range (string): what you consider the typical size range for this object in meters
- suggested_scale (float or null): if is_proper is false, provide a new uniform scale that would make the object size proper; otherwise null

Guidelines for judgment:
- Consider the typical size range of the object category in the real world
- Allow for reasonable variation within categories, but make sure the variation range is not too big
- Focus on both the longest dimension and overall proportions
- If any dimension seems extremely unrealistic (>5x or <0.2x typical), mark as improper

Scale calculation guidelines:
- suggested_scale = target_longest_dimension_m / (current_longest_dimension_in_model_units)
- Target the midpoint of the typical size range for the longest dimension
- Round suggested_scale to 3 decimal places
- Apply uniform scale across all axes

Formatting rules:
- Output JSON only; no extra text
- Use boolean true/false (not strings)
- If is_proper is true, set assessment and suggested_scale to null

Example outputs:
{
  "is_proper": true,
  "assessment": null,
  "typical_size_range": "0.08–0.12 m height, 0.07–0.10 m diameter",
  "suggested_scale": null
}

{
  "is_proper": false,
  "assessment": "too_big",
  "typical_size_range": "0.08–0.12 m height, 0.07–0.10 m diameter",
  "suggested_scale": 0.079
}
"""

client = OpenAI()  # 依赖 OPENAI_API_KEY

def is_gpt5(model: str) -> bool:
    return (model or "").lower().startswith("gpt-5")

def token_kw(model: str, n: int):
    """
    为不同模型返回正确的 token 上限参数
    """
    m = (model or "").lower()
    if m.startswith("gpt-5"):
        return {"max_completion_tokens": n}
    return {"max_tokens": n}

def temp_kw(model: str, t: float):
    m = (model or "").lower()
    # gpt-5 系列不传或固定 1
    if m.startswith("gpt-5"):
        return {}  # 或者 return {"temperature": 1}
    return {"temperature": t}

def build_validation_message(object_name: str, category: str, dimension: str, scale: List[float], real_size: str) -> List[Dict[str, Any]]:
    """
    构造尺寸验证消息，包含 dimension 和 scale 信息
    """
    # 解析 real_size 获取最大尺寸用于参考
    dimension_info = ""
    try:
        real_dims = [float(x) for x in real_size.split('*')]
        max_real = max(real_dims)
        dimension_info = f"(max real dimension: {max_real:.2f}cm)"
    except Exception:
        pass

    # 解析 dimension（模型单位）获取最大维度
    model_max_info = ""
    try:
        model_dims = [float(x) for x in dimension.split('*')]
        max_model = max(model_dims)
        model_max_info = f"(max model dimension: {max_model:.4f})"
    except Exception:
        pass

    # 规范 scale 展示
    try:
        scale_str = "[" + ", ".join(f"{float(s):.6g}" for s in scale) + "]"
    except Exception:
        scale_str = str(scale)

    text = f'''Object validation request:
- object_name: "{object_name}"
- category: "{category}"
- dimension: "{dimension}" {model_max_info}
- scale: {scale_str}
- real_size: "{real_size}" {dimension_info}

Please validate if this size is reasonable for this object in the real world.
Return JSON only.'''

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]


def request_validation(model: str, messages: List[Dict[str, Any]], *, json_mode: bool, timeout: int, max_tokens: int) -> str:
    """
    调用 LLM 进行尺寸验证
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            **({"response_format": {"type": "json_object"}} if json_mode else {}),
            **temp_kw(model, 0.1),  # 低温度保证一致性
            **token_kw(model, max_tokens),
            timeout=timeout,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}")

def call_llm_validation(
    object_name: str,
    category: str,
    dimension: str,
    scale: List[float],
    real_size: str,
    model: str,
    timeout: int = 60
) -> dict:
    """
    调用 LLM 验证对象尺寸（包含 suggested_scale 字段校验）
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            messages = build_validation_message(object_name, category, dimension, scale, real_size)
            text = request_validation(
                model, messages,
                json_mode=not is_gpt5(model),
                timeout=timeout,
                max_tokens=200,
            )

            # 解析 JSON 响应
            try:
                result = json.loads(text)

                # 必填字段（包含 suggested_scale）
                required_fields = ["is_proper", "assessment", "typical_size_range", "suggested_scale"]
                if not all(field in result for field in required_fields):
                    raise ValueError(f"Missing required fields in response: {text}")

                # 类型约束
                if not isinstance(result["is_proper"], bool):
                    raise ValueError(f"is_proper must be boolean, got: {type(result['is_proper'])}")

                if result["is_proper"] is True:
                    # 统一取值（强制 JSON 一致性）
                    result["assessment"] = None
                    result["suggested_scale"] = None
                else:
                    # 不合理时：assessment 必须为 too_big/too_small，且 suggested_scale 必须为浮点数
                    if result.get("assessment") not in ("too_big", "too_small"):
                        raise ValueError(
                            f"assessment must be 'too_big' or 'too_small' when is_proper=false; got: {result.get('assessment')}"
                        )
                    ss = result.get("suggested_scale", None)
                    if ss is None:
                        raise ValueError("suggested_scale is required when is_proper=false")
                    try:
                        result["suggested_scale"] = float(ss)
                    except Exception:
                        raise ValueError(f"suggested_scale must be a float when is_proper=false; got: {repr(ss)}")

                return result

            except json.JSONDecodeError:
                # 尝试从文本中提取 JSON
                json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))

                    # 同样的必填与一致性校验
                    required_fields = ["is_proper", "assessment", "reason", "typical_size_range", "suggested_scale"]
                    if not all(field in result for field in required_fields):
                        raise ValueError(f"Missing required fields in response: {text}")

                    if not isinstance(result["is_proper"], bool):
                        raise ValueError(f"is_proper must be boolean, got: {type(result['is_proper'])}")

                    if result["is_proper"] is True:
                        result["assessment"] = None
                        result["suggested_scale"] = None
                    else:
                        if result.get("assessment") not in ("too_big", "too_small"):
                            raise ValueError(
                                f"assessment must be 'too_big' or 'too_small' when is_proper=false; got: {result.get('assessment')}"
                            )
                        ss = result.get("suggested_scale", None)
                        if ss is None:
                            raise ValueError("suggested_scale is required when is_proper=false")
                        try:
                            result["suggested_scale"] = float(ss)
                        except Exception:
                            raise ValueError(f"suggested_scale must be a float when is_proper=false; got: {repr(ss)}")

                    return result

                raise ValueError(f"Cannot parse JSON from response: {text}")

        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                wait_time = (2 ** (attempt - 1)) + random.random()
                time.sleep(wait_time)

    raise RuntimeError(f"All attempts failed. Last error: {last_err}")


def postprocess_validation(record: dict, uuid: str, object_data: dict) -> Tuple[str, dict]:
    """
    后处理验证结果，返回 (uuid, validation_data)
    """
    if not isinstance(record, dict):
        return uuid, record

    # 强制一致性：is_proper=true 时 assessment/suggested_scale 必须为 null
    if record.get("is_proper") is True:
        record["assessment"] = None
        record["suggested_scale"] = None

    # 只返回验证相关的字段（包含 suggested_scale）
    validation_data = {
        "is_proper": record.get("is_proper"),
        "assessment": record.get("assessment"),
        "reason": record.get("reason", ""),
        "typical_size_range": record.get("typical_size_range", ""),
        "suggested_scale": record.get("suggested_scale", None),
    }

    return uuid, validation_data


def probe_model(model: str):
    """
    测试模型可用性
    """
    try:
        messages = [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": 'Reply with {"test": true} exactly.'}
        ]
        
        text = request_validation(
            model, messages,
            json_mode=not is_gpt5(model),
            timeout=30,
            max_tokens=50
        )
        
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and obj.get("test") is True:
                return True, None
        except:
            if '"test":true' in (text or "").replace(" ", "").lower():
                return True, None
                
        return False, f"Unexpected response: {repr(text)[:200]}"
    except Exception as e:
        return False, str(e)

def run_validation_batch(input_path: Path, output_path: Path, model: str, limit: int = None, workers: int = MAX_WORKERS, dry_run: bool = False):
    """
    批量验证对象尺寸
    """
    # 读取输入数据
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    items = sorted(input_data.items(), key=lambda kv: kv[0])
    if limit is not None and limit > 0:
        items = items[:limit]

    # 模型可用性检查
    ok, err = probe_model(model)
    if not ok:
        print(f"[Model probe] model={model} failed: {err}")
        print("Tip: try --model gpt-5-mini / gpt-4o / gpt-4.1 / gpt-4o-mini")
        return

    print(f"Model {model} is ready. Processing {len(items)} objects...")

    results: Dict[str, Dict[str, Any]] = {}  # {uuid: validation_data}
    errors: Dict[str, str] = {}

    # 干跑模式
    if dry_run:
        print("[Dry-run] Validating the first 3 objects...")
        for uuid, obj_data in items[:3]:
            object_name = obj_data.get("object_name", "")
            category = obj_data.get("category", "")
            dimension = obj_data.get("dimension", "")
            scale = obj_data.get("scale", [])
            real_size = obj_data.get("real_size", "")

            try:
                validation_result = call_llm_validation(
                    object_name, category, dimension, scale, real_size, model=model
                )
                uuid_key, validation_data = postprocess_validation(validation_result, uuid, obj_data)
                print(f"{uuid_key} => {json.dumps(validation_data, ensure_ascii=False, indent=2)}")
            except Exception as e:
                print(f"{uuid} ERROR: {repr(e)}")
        return

    # 批量处理
    bar = tqdm(total=len(items), desc="Validating sizes", unit="obj") if tqdm else None

    def work(item: Tuple[str, Dict[str, Any]]):
        uuid, obj_data = item
        object_name = obj_data.get("object_name", "")
        category = obj_data.get("category", "")
        dimension = obj_data.get("dimension", "")
        scale = obj_data.get("scale", [])
        real_size = obj_data.get("real_size", "")

        validation_result = call_llm_validation(
            object_name, category, dimension, scale, real_size, model=model
        )
        uuid_key, validation_data = postprocess_validation(validation_result, uuid, obj_data)
        return uuid_key, validation_data

    with ThreadPoolExecutor(max_workers=workers) as executor:
        fut2uuid = {}
        for item in items:
            fut = executor.submit(work, item)
            fut2uuid[fut] = item[0]

        for fut in as_completed(fut2uuid):
            uuid = fut2uuid[fut]
            try:
                uuid_key, validation_data = fut.result()
                results[uuid_key] = validation_data
            except Exception as e:
                errors[uuid] = repr(e)

            if bar:
                bar.update(1)
                if errors and len(errors) % 20 == 0:
                    bar.set_postfix(errors=len(errors))

    if bar:
        bar.close()

    # 统计信息
    proper_count = len([r for r in results.values() if r.get("is_proper") is True])
    improper_count = len([r for r in results.values() if r.get("is_proper") is False])
    too_big_count = len([r for r in results.values() if r.get("assessment") == "too_big"])
    too_small_count = len([r for r in results.values() if r.get("assessment") == "too_small"])

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nValidation complete!")
    print(f"Results saved to: {output_path}")
    print(f"Successfully processed: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Proper sizes: {proper_count}")
    print(f"Improper sizes: {improper_count}")
    print(f"  - Too big: {too_big_count}")
    print(f"  - Too small: {too_small_count}")

    if errors:
        error_file = output_path.with_name(output_path.stem + "_errors.json")
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump({"count": len(errors), "errors": errors}, f, ensure_ascii=False, indent=2)
        print(f"Errors saved to: {error_file}")

    # 也创建一个包含统计信息的文件
    summary_file = output_path.with_name(output_path.stem + "_summary.json")
    summary_data = {
        "total_processed": len(results),
        "total_errors": len(errors),
        "proper_sizes": proper_count,
        "improper_sizes": improper_count,
        "too_big": too_big_count,
        "too_small": too_small_count,
        "percentage_proper": round(proper_count / len(results) * 100, 2) if results else 0
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Validate real sizes of RobotWin objects using LLM')
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="robotwin_real_sizes_meter.json",
        help="Input JSON file with real sizes (default: robotwin_real_sizes.json)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str, 
        default="size_validation_results.json",
        help="Output JSON file for validation results (default: size_validation_results.json)"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Max concurrent workers (default: {MAX_WORKERS})"
    )
    
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Only process first N objects"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run first 3 items and print results"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("RobotWin Objects Size Validation Tool")
    print("=" * 70)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTION'}")
    if args.limit:
        print(f"Limit: {args.limit} objects")
    print("=" * 70)
    
    run_validation_batch(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model=args.model,
        limit=args.limit,
        workers=args.workers,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()