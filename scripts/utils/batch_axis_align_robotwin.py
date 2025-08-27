#!/usr/bin/env python3
"""
RobotWin 数据集批量坐标轴对齐脚本
===============================
针对RobotWin数据集中的简化对象进行批量坐标轴转换处理。

简化对象定义：同时包含 visual.glb, collision.glb 和 info.json 的目录

功能：
- 自动识别简化对象目录
- 对visual.glb和collision.glb进行坐标轴转换
- 支持试跑模式（不实际处理文件）
- 支持限制处理数量
- 详细的进度显示和错误处理

使用示例：
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


class RobotWinAxisAligner:
    def __init__(self, robotwin_dir, blender_path="blender", axis_align_script=None):
        self.robotwin_dir = Path(robotwin_dir)
        self.blender_path = blender_path
        
        # 查找axis_align.py脚本
        if axis_align_script:
            self.axis_align_script = Path(axis_align_script)
        else:
            # 尝试在当前目录查找
            script_candidates = [
                Path(__file__).parent / "axis_align.py",
                Path("axis_align.py"),
                Path("./axis_align.py")
            ]
            self.axis_align_script = None
            for candidate in script_candidates:
                if candidate.exists():
                    self.axis_align_script = candidate
                    break
        
        if not self.axis_align_script or not self.axis_align_script.exists():
            raise FileNotFoundError("未找到axis_align.py脚本，请使用 --axis-align-script 指定路径")
        
        # 设置日志
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志记录"""
        log_file = Path(__file__).parent / f"robotwin_align_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def is_simplified_object(self, directory):
        """
        判断是否为简化对象目录
        
        Args:
            directory (Path): 目录路径
            
        Returns:
            bool: 如果同时包含visual.glb, collision.glb和info.json则为True
        """
        required_files = ['visual.glb', 'collision.glb', 'info.json']
        return all((directory / filename).exists() for filename in required_files)
    
    def find_simplified_objects(self):
        """
        查找所有简化对象目录
        
        Returns:
            list: 简化对象目录路径列表
        """
        if not self.robotwin_dir.exists():
            raise FileNotFoundError(f"RobotWin目录不存在: {self.robotwin_dir}")
        
        simplified_objects = []
        
        for item in self.robotwin_dir.iterdir():
            if item.is_dir() and self.is_simplified_object(item):
                simplified_objects.append(item)
        
        # 按UUID排序
        simplified_objects.sort(key=lambda x: x.name)
        return simplified_objects
    
    def run_blender_align(self, input_file, output_dir, axis="X", angle=-90.0, export_format="glb"):
        """
        运行Blender坐标轴对齐
        
        Args:
            input_file (Path): 输入GLB文件
            output_dir (Path): 输出目录
            axis (str): 旋转轴
            angle (float): 旋转角度
            export_format (str): 导出格式
            
        Returns:
            bool: 是否成功
        """
        cmd = [
            self.blender_path,
            "-b",  # 后台模式
            "-P", str(self.axis_align_script),
            "--",
            "--in", str(input_file),
            "--out", str(output_dir),
            "--axis", axis,
            "--angle", str(angle),
            "--format", export_format
        ]
        
        try:
            self.logger.debug(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
                check=False
            )
            
            if result.returncode == 0:
                self.logger.debug(f"Blender处理成功: {input_file.name}")
                return True
            else:
                self.logger.error(f"Blender处理失败: {input_file.name}")
                self.logger.error(f"错误输出: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Blender处理超时: {input_file.name}")
            return False
        except Exception as e:
            self.logger.error(f"执行Blender命令时出错: {e}")
            return False
    
    def process_simplified_object(self, obj_dir, axis="X", angle=-90.0, dry_run=False):
        """
        处理单个简化对象目录
        
        Args:
            obj_dir (Path): 简化对象目录
            axis (str): 旋转轴
            angle (float): 旋转角度
            dry_run (bool): 是否为试跑模式
            
        Returns:
            dict: 处理结果统计
        """
        result = {
            'uuid': obj_dir.name,
            'visual_success': False,
            'collision_success': False,
            'error': None
        }
        
        visual_file = obj_dir / 'visual.glb'
        collision_file = obj_dir / 'collision.glb'
        
        if dry_run:
            self.logger.info(f"[试跑] 将处理 {obj_dir.name}")
            self.logger.info(f"  - visual.glb: 绕{axis}轴旋转{angle}度")
            self.logger.info(f"  - collision.glb: 绕{axis}轴旋转{angle}度")
            result['visual_success'] = True
            result['collision_success'] = True
            return result
        
        try:
            # 创建临时输出目录
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # 处理visual.glb
                self.logger.info(f"处理 {obj_dir.name}/visual.glb...")
                if self.run_blender_align(visual_file, temp_path, axis, angle):
                    # 替换原文件
                    aligned_visual = temp_path / "visual.glb"
                    if aligned_visual.exists():
                        shutil.move(str(aligned_visual), str(visual_file))
                        result['visual_success'] = True
                        self.logger.info(f"✓ visual.glb 处理完成")
                    else:
                        self.logger.error(f"✗ 未找到处理后的visual.glb文件")
                
                # 处理collision.glb
                self.logger.info(f"处理 {obj_dir.name}/collision.glb...")
                if self.run_blender_align(collision_file, temp_path, axis, angle):
                    # 替换原文件
                    aligned_collision = temp_path / "collision.glb"
                    if aligned_collision.exists():
                        shutil.move(str(aligned_collision), str(collision_file))
                        result['collision_success'] = True
                        self.logger.info(f"✓ collision.glb 处理完成")
                    else:
                        self.logger.error(f"✗ 未找到处理后的collision.glb文件")
        
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"处理 {obj_dir.name} 时出错: {e}")
        
        return result
    
    def process_all(self, axis="X", angle=-90.0, limit=None, dry_run=False):
        """
        处理所有简化对象
        
        Args:
            axis (str): 旋转轴
            angle (float): 旋转角度  
            limit (int): 限制处理数量
            dry_run (bool): 是否为试跑模式
            
        Returns:
            dict: 处理统计结果
        """
        simplified_objects = self.find_simplified_objects()
        
        if not simplified_objects:
            self.logger.warning("未找到任何简化对象")
            return {'total': 0, 'processed': 0, 'success': 0, 'failed': 0}
        
        # 应用数量限制
        if limit:
            simplified_objects = simplified_objects[:limit]
        
        stats = {
            'total': len(simplified_objects),
            'processed': 0,
            'success': 0,
            'failed': 0,
            'visual_success': 0,
            'collision_success': 0,
            'errors': []
        }
        
        self.logger.info("=" * 80)
        self.logger.info(f"开始处理RobotWin简化对象")
        self.logger.info(f"模式: {'试跑模式' if dry_run else '执行模式'}")
        self.logger.info(f"旋转参数: 绕{axis}轴旋转{angle}度")
        self.logger.info(f"处理数量: {len(simplified_objects)} / {len(self.find_simplified_objects())} 个简化对象")
        self.logger.info("=" * 80)
        
        for i, obj_dir in enumerate(simplified_objects, 1):
            self.logger.info(f"\n[{i}/{len(simplified_objects)}] 处理 {obj_dir.name}")
            
            result = self.process_simplified_object(obj_dir, axis, angle, dry_run)
            stats['processed'] += 1
            
            if result['visual_success']:
                stats['visual_success'] += 1
            if result['collision_success']:
                stats['collision_success'] += 1
            
            if result['visual_success'] and result['collision_success']:
                stats['success'] += 1
            else:
                stats['failed'] += 1
                if result['error']:
                    stats['errors'].append(f"{obj_dir.name}: {result['error']}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='RobotWin数据集批量坐标轴对齐')
    
    parser.add_argument(
        '--robotwin-dir',
        default='D:/codefield/VLA/objaverse/robotwin_objects/robotwin_objects',
        help='RobotWin数据集目录路径'
    )
    
    parser.add_argument(
        '--blender-path',
        default='blender',
        help='Blender可执行文件路径 (默认从PATH查找)'
    )
    
    parser.add_argument(
        '--axis-align-script',
        help='axis_align.py脚本路径 (默认在当前目录查找)'
    )
    
    parser.add_argument(
        '--axis',
        choices=['X', 'Y', 'Z'],
        default='X',
        help='旋转轴 (默认: X)'
    )
    
    parser.add_argument(
        '--angle',
        type=float,
        default=-90.0,
        help='旋转角度 (默认: -90.0)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='限制处理的简化对象数量'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='试跑模式，不实际处理文件'
    )
    
    args = parser.parse_args()
    
    try:
        # 检查Blender是否可用
        result = subprocess.run([args.blender_path, '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print(f"错误: 无法运行Blender: {args.blender_path}")
            print("请安装Blender或使用 --blender-path 指定正确路径")
            return 1
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"错误: 无法找到或运行Blender: {args.blender_path}")
        return 1
    
    try:
        aligner = RobotWinAxisAligner(
            robotwin_dir=args.robotwin_dir,
            blender_path=args.blender_path,
            axis_align_script=args.axis_align_script
        )
        
        stats = aligner.process_all(
            axis=args.axis,
            angle=args.angle,
            limit=args.limit,
            dry_run=args.dry_run
        )
        
        # 打印最终统计
        print("\n" + "=" * 60)
        print("处理完成统计:")
        print(f"总计简化对象: {stats['total']}")
        print(f"已处理对象: {stats['processed']}")
        print(f"完全成功: {stats['success']}")
        print(f"部分/完全失败: {stats['failed']}")
        print(f"Visual GLB成功: {stats['visual_success']}")
        print(f"Collision GLB成功: {stats['collision_success']}")
        
        if stats['errors']:
            print(f"\n错误详情:")
            for error in stats['errors'][:5]:  # 只显示前5个错误
                print(f"  - {error}")
            if len(stats['errors']) > 5:
                print(f"  ... 还有{len(stats['errors']) - 5}个错误")
        
        if args.dry_run:
            print(f"\n提示: 这是试跑模式。要执行实际处理，请去掉 --dry-run 参数")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())