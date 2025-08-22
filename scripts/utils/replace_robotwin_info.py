import json
import os
import argparse
from typing import Optional

# 加载 robotwin_info_generated_by_llm_proceed_scale.json 文件
def load_robotwin_info(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 存储每个 uuid 对应的 info.json
def save_info_to_file(uuid: str, data: dict, base_path: str, dry_run: bool = False):
    directory = os.path.join(base_path, uuid)
    info_file_path = os.path.join(directory, 'info.json')

    if dry_run:
        # 仅展示将要执行的操作
        need_mkdir = not os.path.exists(directory)
        print(f"[DRY-RUN] {'将创建目录' if need_mkdir else '目录已存在'}: {directory}")
        print(f"[DRY-RUN] 将写入文件: {info_file_path}")
        return

    # 实际执行
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(info_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 处理每个 uuid
def process_robotwin_data(input_json: str, base_path: str, limit: Optional[int] = None, dry_run: bool = False):
    robotwin_data = load_robotwin_info(input_json)

    # 将 limit 规范化：None 或 <=0 视为不限制
    total = len(robotwin_data)
    effective_limit = None if (limit is None or limit <= 0) else int(limit)

    # 迭代数据（Python 3.7+ 字典保持插入顺序；若需特定顺序可自行排序）
    processed = 0
    for uuid, obj in robotwin_data.items():
        if effective_limit is not None and processed >= effective_limit:
            break
        save_info_to_file(uuid, obj, base_path, dry_run=dry_run)
        processed += 1

    action = "计划" if dry_run else "已"
    scope = f"（limit={effective_limit}）" if effective_limit is not None else "（全部）"
    print(f"{action}处理 {processed}/{total} 个条目 {scope}。目标目录：{base_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="按 uuid 拆分 JSON 并写入各自 info.json")
    parser.add_argument(
        "--input",
        default="updated_robotwin_info.json",
        help="输入 JSON 文件路径（默认：updated_robotwin_info.json）",
    )
    parser.add_argument(
        "--base-path",
        default=r"D:\codefield\VLA\objaverse\robotwin_objects\robotwin_objects",
        help=r"输出基础目录路径（默认：D:\codefield\VLA\objaverse\robotwin_objects\robotwin_objects）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 个 uuid（默认：不限制）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅演练：打印将要进行的操作，不创建目录、不写文件",
    )
    return parser.parse_args()

# 程序入口
def main():
    args = parse_args()
    process_robotwin_data(
        input_json=args.input,
        base_path=args.base_path,
        limit=args.limit,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()
