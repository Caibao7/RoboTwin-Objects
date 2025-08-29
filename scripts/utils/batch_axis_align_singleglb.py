#!/usr/bin/env python3
"""
RobotWin 数据集批量坐标轴对齐脚本（单 GLB 版）
==========================================
针对 RobotWin 数据集，每个 UUID 目录下只有一个 .glb 文件的场景，批量做坐标轴旋转。

变更要点：
- 不再要求 visual.glb / collision.glb / info.json 同时存在
- 每个对象目录仅当且仅当“恰好一个 .glb 文件”时才会被处理
- 直接对该唯一 .glb 执行轴对齐并原地覆盖

用法示例：
python batch_axis_align_robotwin.py --dry-run --limit 5
python batch_axis_align_robotwin.py --axis X --angle -90 --limit 10
python batch_axis_align_robotwin.py --blender-path "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe"
"""

import os
import sys
import subprocess
import argparse
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Optional


class RobotWinAxisAligner:
    def __init__(self, robotwin_dir, blender_path="blender", axis_align_script=None):
        self.robotwin_dir = Path(robotwin_dir)
        self.blender_path = blender_path

        # 查找 axis_align.py 脚本
        if axis_align_script:
            self.axis_align_script = Path(axis_align_script)
        else:
            script_candidates = [
                Path(__file__).parent / "axis_align.py",
                Path("axis_align.py"),
                Path("./axis_align.py"),
            ]
            self.axis_align_script = None
            for c in script_candidates:
                if c.exists():
                    self.axis_align_script = c
                    break

        if not self.axis_align_script or not self.axis_align_script.exists():
            raise FileNotFoundError("未找到 axis_align.py 脚本，请使用 --axis-align-script 指定路径")

        # 设置日志
        self._setup_logging()

    def _setup_logging(self):
        """设置日志记录"""
        log_file = Path(__file__).parent / f"robotwin_align_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    # ---------- 新的对象判定逻辑：恰好一个 .glb ----------

    def _list_glbs_in_dir(self, directory: Path) -> List[Path]:
        """仅列举该目录第一层中的 .glb 文件（不递归）"""
        return [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".glb"]

    def is_single_glb_object(self, directory: Path) -> bool:
        """
        判断目录是否为“单 glb 对象目录”
        条件：目录下（非递归）恰好一个 .glb 文件
        """
        if not directory.is_dir():
            return False
        glbs = self._list_glbs_in_dir(directory)
        return len(glbs) == 1

    def find_single_glb_objects(self) -> List[Path]:
        """
        查找根目录下一层中，恰好包含一个 .glb 文件的目录
        """
        if not self.robotwin_dir.exists():
            raise FileNotFoundError(f"RobotWin 目录不存在: {self.robotwin_dir}")

        objs = []
        for item in self.robotwin_dir.iterdir():
            if item.is_dir() and self.is_single_glb_object(item):
                objs.append(item)

        # 按目录名（通常为 UUID）排序，保证可重复性
        objs.sort(key=lambda x: x.name)
        return objs

    # ---------- 调用 Blender 的封装 ----------

    def run_blender_align(self, input_file: Path, output_dir: Path, axis: str = "X", angle: float = -90.0, export_format: str = "glb") -> bool:
        """
        运行 Blender 进行坐标轴对齐
        约定：axis_align.py 会把处理后的文件输出到 --out 指向的目录，文件名与输入文件同名
        """
        cmd = [
            self.blender_path,
            "-b",  # 后台模式
            "-P",
            str(self.axis_align_script),
            "--",
            "--in",
            str(input_file),
            "--out",
            str(output_dir),
            "--axis",
            axis,
            "--angle",
            str(angle),
            "--format",
            export_format,
        ]

        try:
            self.logger.debug(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)

            if result.returncode == 0:
                self.logger.debug(f"Blender 处理成功: {input_file.name}")
                return True
            else:
                self.logger.error(f"Blender 处理失败: {input_file.name}")
                if result.stderr:
                    self.logger.error(f"错误输出: {result.stderr.strip()}")
                if result.stdout:
                    self.logger.info(f"标准输出: {result.stdout.strip()}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error(f"Blender 处理超时: {input_file.name}")
            return False
        except Exception as e:
            self.logger.error(f"执行 Blender 命令时出错: {e}")
            return False

    # ---------- 处理单对象目录 ----------

    def process_single_glb_object(self, obj_dir: Path, axis: str = "X", angle: float = -90.0, dry_run: bool = False) -> dict:
        """
        处理一个只包含单个 .glb 的对象目录
        返回结果结构：
            {
                'uuid': <目录名>,
                'glb_name': <文件名>,
                'success': bool,
                'error': Optional[str],
            }
        """
        glbs = self._list_glbs_in_dir(obj_dir)
        result = {
            "uuid": obj_dir.name,
            "glb_name": glbs[0].name if len(glbs) == 1 else None,
            "success": False,
            "error": None,
        }

        # 目录不满足条件时，给出明确错误
        if len(glbs) == 0:
            msg = "目录中未找到 .glb 文件"
            self.logger.warning(f"{obj_dir.name}: {msg}")
            result["error"] = msg
            return result
        if len(glbs) > 1:
            msg = f"目录中存在多个 .glb 文件（{len(glbs)} 个），为避免歧义已跳过"
            self.logger.warning(f"{obj_dir.name}: {msg}")
            result["error"] = msg
            return result

        glb_file = glbs[0]

        if dry_run:
            self.logger.info(f"[试跑] 将处理 {obj_dir.name} -> {glb_file.name}，绕 {axis} 轴旋转 {angle} 度")
            result["success"] = True
            return result

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                self.logger.info(f"处理 {obj_dir.name}/{glb_file.name} ...")
                ok = self.run_blender_align(glb_file, temp_path, axis, angle, export_format="glb")
                if not ok:
                    result["error"] = "Blender 子进程返回失败"
                    return result

                # 约定：输出文件名与输入同名
                aligned_path = temp_path / glb_file.name
                if not aligned_path.exists():
                    # 兼容某些 axis_align.py 把扩展名小写变更或只写 .glb 的情况（通常相同）
                    # 若需要更鲁棒，可在此枚举 temp_dir 内所有 .glb 并尝试按“同名或仅扩展名不同”匹配
                    msg = f"未找到处理后的文件：{aligned_path.name}"
                    self.logger.error(msg)
                    result["error"] = msg
                    return result

                # 覆盖原文件
                shutil.move(str(aligned_path), str(glb_file))
                self.logger.info(f"✓ 已覆盖输出：{glb_file.name}")
                result["success"] = True
                return result

        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"处理 {obj_dir.name} 时出错: {e}")
            return result

    # ---------- 批量处理 ----------

    def process_all(self, axis: str = "X", angle: float = -90.0, limit: Optional[int] = None, dry_run: bool = False) -> dict:
        """
        批量处理所有“恰好一个 .glb 文件”的对象目录
        返回统计信息：
            {
                'total': int,
                'processed': int,
                'success': int,
                'failed': int,
                'errors': List[str]
            }
        """
        all_candidates = [d for d in self.robotwin_dir.iterdir() if d.is_dir()]
        single_glb_objects = self.find_single_glb_objects()

        if not all_candidates:
            self.logger.warning("根目录下未发现任何子目录。")
        if not single_glb_objects:
            self.logger.warning("未发现满足“恰好一个 .glb”的对象目录。")

        # 应用数量限制
        original_count = len(single_glb_objects)
        if limit:
            single_glb_objects = single_glb_objects[:limit]

        stats = {
            "total": len(single_glb_objects),
            "processed": 0,
            "success": 0,
            "failed": 0,
            "errors": [],
        }

        self.logger.info("=" * 80)
        self.logger.info("开始处理 RobotWin 单 GLB 对象")
        self.logger.info(f"模式: {'试跑模式' if dry_run else '执行模式'}")
        self.logger.info(f"旋转参数: 绕 {axis} 轴旋转 {angle} 度")
        self.logger.info(f"满足条件的对象目录: {len(single_glb_objects)} / {original_count}（在所有子目录中）")
        self.logger.info("=" * 80)

        for i, obj_dir in enumerate(single_glb_objects, 1):
            self.logger.info(f"\n[{i}/{len(single_glb_objects)}] 处理 {obj_dir.name}")
            result = self.process_single_glb_object(obj_dir, axis=axis, angle=angle, dry_run=dry_run)

            stats["processed"] += 1
            if result["success"]:
                stats["success"] += 1
            else:
                stats["failed"] += 1
                if result.get("error"):
                    stats["errors"].append(f"{obj_dir.name}: {result['error']}")

        return stats


def main():
    parser = argparse.ArgumentParser(description="RobotWin 数据集批量坐标轴对齐（单 GLB 版）")

    parser.add_argument(
        "--robotwin-dir",
        default=r"D:\codefield\VLA\Task-Scalinggg\Objects-Dataset\dining_room_objects",
        help="Sketchfab 数据集根目录",
    )

    parser.add_argument(
        "--blender-path",
        default="blender",
        help="Blender 可执行文件路径 (默认从 PATH 查找)",
    )

    parser.add_argument(
        "--axis-align-script",
        default=r"D:\codefield\VLA\objaverse\RoboTwin-Objects\scripts\utils\axis_align.py", 
        help="axis_align.py 脚本路径 (默认在当前目录查找)",
    )

    parser.add_argument(
        "--axis",
        choices=["X", "Y", "Z"],
        default="X",
        help="旋转轴 (默认: X)",
    )

    parser.add_argument(
        "--angle",
        type=float,
        default=0,
        help="旋转角度 (默认: 0)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="限制处理的对象数量（按目录名排序后取前 N）",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试跑模式，不实际处理文件",
    )

    args = parser.parse_args()

    # 先检查 Blender 是否可用
    try:
        result = subprocess.run([args.blender_path, "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print(f"错误: 无法运行 Blender: {args.blender_path}")
            print("请安装 Blender 或使用 --blender-path 指定正确路径")
            return 1
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"错误: 无法找到或运行 Blender: {args.blender_path}")
        return 1

    try:
        aligner = RobotWinAxisAligner(
            robotwin_dir=args.robotwin_dir,
            blender_path=args.blender_path,
            axis_align_script=args.axis_align_script,
        )

        stats = aligner.process_all(axis=args.axis, angle=args.angle, limit=args.limit, dry_run=args.dry_run)

        # 打印最终统计
        print("\n" + "=" * 60)
        print("处理完成统计:")
        print(f"满足条件的对象目录: {stats['total']}")
        print(f"已处理对象: {stats['processed']}")
        print(f"成功: {stats['success']}")
        print(f"失败: {stats['failed']}")

        if stats["errors"]:
            print(f"\n错误详情:")
            for error in stats["errors"][:5]:  # 只显示前 5 个错误
                print(f"  - {error}")
            if len(stats["errors"]) > 5:
                print(f"  ... 还有 {len(stats['errors']) - 5} 个错误")

        if args.dry_run:
            print(f"\n提示: 这是试跑模式。要执行实际处理，请去掉 --dry-run 参数")

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
