# -*- coding: utf-8 -*-
"""
只同步 uuid/info.json（不再删除任何远端文件）
- include: 仅 uuid/info.json
"""
import os, re, json, time, random
from pathlib import Path
import huggingface_hub
from huggingface_hub import upload_folder
try:
    from huggingface_hub.errors import HfHubHTTPError  # 新版
except Exception:
    HfHubHTTPError = Exception  # 旧版兜底

# ===== 配置区 =====
REPO = "Task-Scalinggg/Objects-Dataset"
LOCAL_ROOT = Path(r"D:\codefield\VLA\objaverse\robotwin_objects\robotwin_objects")  # 含 UUID 子目录
OBJECTS_JSON = LOCAL_ROOT / "robotwin_objects.json"

GROUP_SIZE = 30            # 每次提交的 UUID 数量
MAX_RETRIES = 4            # 单次提交最大重试次数
BASE_BACKOFF = 5           # 指数退避起始秒
SLEEP_BETWEEN_COMMITS = 5  # 提交之间的固定等待（秒）
DRY_RUN = False            # 先改 True 预览；确认后改回 False 执行
# ==================

# 传输加速（装了 hf-transfer 则自动启用）
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

def load_objects_json(path: Path):
    assert path.exists(), f"找不到 objects.json: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # {uuid: {...}, ...}

def pick_uuids_to_push(root: Path, objects_meta: dict):
    """只挑选本地存在 uuid/info.json 的目录；返回 [(uuid, uuid_dir)]"""
    todo = []
    for uuid in objects_meta.keys():
        if not UUID_RE.match(uuid):
            continue
        uuid_dir = root / uuid
        if not uuid_dir.is_dir():
            continue
        if not (uuid_dir / "info.json").exists():
            continue
        todo.append((uuid, uuid_dir))
    todo.sort(key=lambda x: x[0])
    return todo

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _upload_group(group_items):
    """
    group_items: list of (uuid, uuid_dir)
    include: 仅 uuid/info.json
    """
    include_patterns = [f"{uuid}/info.json" for (uuid, _) in group_items]
    kw = dict(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=str(LOCAL_ROOT),
        path_in_repo="robotwin_objects",
        commit_message=f"sync info.json: {group_items[0][0]}..{group_items[-1][0]} ({len(group_items)} uuids)",
    )

    if DRY_RUN:
        print(f"[DRY] include {len(include_patterns)} files")
        print("      sample:", include_patterns[:3])
        return True

    # 新版参数（include/exclude）；兼容旧版（allow/ignore）
    try:
        upload_folder(**kw, include=include_patterns, exclude=[".git/*"])
    except TypeError:
        upload_folder(**kw, allow_patterns=include_patterns, ignore_patterns=[".git/*"])
    return True

def upload_group_with_retries(group_items):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Uploading group ({len(group_items)}): {group_items[0][0]} .. {group_items[-1][0]} (attempt {attempt})")
            ok = _upload_group(group_items)
            if ok:
                print("OK group")
                return True
        except HfHubHTTPError as e:
            msg = str(e)
            if ("429" in msg or "Too Many Requests" in msg
                or "403" in msg or "Request has expired" in msg):
                wait = min(BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 3), 120)
                print(f"[WARN] 限流/过期，{wait:.1f}s 后重试...")
                time.sleep(wait)
                continue
            print(f"[ERROR] 失败：{e}")
            return False
        except Exception as e:
            print(f"[ERROR] 异常：{e}")
            return False
    return False

def main():
    print(f"huggingface_hub={huggingface_hub.__version__}")
    print(f"LOCAL_ROOT={LOCAL_ROOT}")
    print(f"DRY_RUN={DRY_RUN}")

    meta = load_objects_json(OBJECTS_JSON)
    todo = pick_uuids_to_push(LOCAL_ROOT, meta)
    print(f"待推送 UUID 数：{len(todo)}（仅含本地存在 info.json 的目录）")
    if not todo:
        return

    for gi, group in enumerate(chunks(todo, GROUP_SIZE), 1):
        ok = upload_group_with_retries(group)
        if not ok:
            print("[WARN] 本组最终失败；重跑脚本会继续补齐。")
        time.sleep(SLEEP_BETWEEN_COMMITS)

if __name__ == "__main__":
    main()
