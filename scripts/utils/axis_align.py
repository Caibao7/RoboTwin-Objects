#!/usr/bin/env python3
"""
Batch Axis Realigner for Blender
================================
Rotate a batch of 3D files by a fixed angle around a specified axis to “hard-fix” their coordinate system,
and export them to a target directory.

Default use case: Convert models where Y is treated as Z by rotating **-90° around the X axis** (Y → Z).

Supported: GLB/GLTF, FBX, OBJ, DAE (COLLADA)
Compatible with Blender 3.x/4.x (OBJ import/export operator name changes handled for 4.x)

Example usage (command line/headless):
blender -b -P batch_axis_realign.py -- \
  --in "/path/input_dir_or_file" \
  --out "/path/output_dir" \
  --axis X \
  --angle -90 \
  --format glb \
  --recursive

./blender -b -P axis_align.py -- \
  --in "/home/user/assets/textured.obj" \
  --out "/home/user/output" \
  --axis X \
  --angle -90 \
  --format glb \
  --recursive

Common cases:
- Y-up → Z-up: --axis X --angle -90
- Z-up → Y-up: --axis X --angle 90
- Left-hand ↔ Right-hand coordinate system conversion is NOT handled in this script.

Notes:
- If the model has armatures/animations, directly changing matrix_world may affect rigging; this script is best for static meshes or assets without complex rigs.
- The glTF exporter automatically applies a Y-up transform; this script applies a hard geometry rotation and leaves exporter defaults unchanged (can override with --gltf-yup).
- Test on backups first.
"""

import bpy
import sys
import os
import math
from mathutils import Matrix
from pathlib import Path
import argparse

SUPPORTED_IMPORT = {".glb", ".gltf", ".fbx", ".obj", ".dae"}
SUPPORTED_EXPORT = {"glb", "gltf", "fbx", "obj", "dae"}

# ---------------------------
# CLI argument parsing
# ---------------------------

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser(description="Batch Axis Realigner for Blender")
    p.add_argument("--in", dest="input_path", required=True, help="Input file or directory")
    p.add_argument("--out", dest="output_dir", required=True, help="Output directory (required even if input is a file)")
    p.add_argument("--axis", choices=["X", "Y", "Z"], default="X", help="World axis to rotate around")
    p.add_argument("--angle", type=float, default=-90.0, help="Rotation angle in degrees (e.g., -90 means -90° around the chosen axis)")
    p.add_argument("--format", dest="export_format", choices=list(SUPPORTED_EXPORT), default="glb", help="Export format")
    p.add_argument("--recursive", action="store_true", help="If input is a directory, search recursively")
    p.add_argument("--keep-hierarchy", action="store_true", help="Attempt to keep object hierarchy on export (if supported by exporter)")
    p.add_argument("--gltf-yup", action="store_true", help="Force Y-up when exporting glTF")
    p.add_argument("--pattern", default="*", help="File search pattern, e.g., *.glb or *.fbx")
    p.add_argument("--apply-scale", action="store_true", help="Apply scaling (by default only rotation is applied)")
    p.add_argument("--no-apply", action="store_true", help="Do not apply transforms with transform_apply (only modify matrix)")
    return p.parse_args(argv)

# ---------------------------
# Scene management
# ---------------------------

def new_clean_file():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if not bpy.data.scenes:
        bpy.data.scenes.new("Scene")
    bpy.context.window.scene = bpy.data.scenes[0]

# ---------------------------
# Import by file extension
# ---------------------------

def import_any(filepath: str):
    ext = Path(filepath).suffix.lower()
    print(f"[Import] {filepath}")
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=filepath)  # Blender 4.x
        except Exception:
            bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == ".dae":
        bpy.ops.wm.collada_import(filepath=filepath)
    else:
        raise RuntimeError(f"Unsupported input extension: {ext}")

# ---------------------------
# Export by format
# ---------------------------

def export_any(filepath_out: str, fmt: str, keep_hierarchy=False, gltf_yup=False):
    fmt = fmt.lower()
    print(f"[Export] {filepath_out}")
    if fmt in {"glb", "gltf"}:
        export_format = "GLB" if fmt == "glb" else "GLTF_SEPARATE"
        bpy.ops.export_scene.gltf(
            filepath=filepath_out,
            export_format=export_format,
            use_selection=False,
            export_extras=True,
            export_yup=gltf_yup,
            export_apply=False,
        )
    elif fmt == "fbx":
        bpy.ops.export_scene.fbx(
            filepath=filepath_out,
            use_selection=False,
            apply_unit_scale=True,
            bake_space_transform=False,
            add_leaf_bones=False,
            use_mesh_modifiers=True,
            path_mode='AUTO'
        )
    elif fmt == "obj":
        try:
            bpy.ops.wm.obj_export(filepath=filepath_out)  # Blender 4.x
        except Exception:
            bpy.ops.export_scene.obj(filepath=filepath_out)
    elif fmt == "dae":
        bpy.ops.wm.collada_export(filepath=filepath_out, selected=False, apply_modifiers=True)
    else:
        raise RuntimeError(f"Unsupported export format: {fmt}")

# ---------------------------
# Geometry rotation & transform application
# ---------------------------

def rotate_objects(new_objects, axis: str = "X", angle_deg: float = -90.0):
    angle = math.radians(angle_deg)
    rot = Matrix.Rotation(angle, 4, axis)
    for obj in new_objects:
        obj.matrix_world = rot @ obj.matrix_world

def apply_transforms(new_objects, apply_rotation=True, apply_scale=False):
    for obj in new_objects:
        if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META", "ARMATURE", "EMPTY", "LIGHT", "CAMERA"}:
            continue
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        try:
            bpy.ops.object.transform_apply(
                location=False,
                rotation=apply_rotation,
                scale=apply_scale,
            )
        except Exception as e:
            print(f"[Warn] transform_apply failed on {obj.name}: {e}")
        finally:
            obj.select_set(False)

# ---------------------------
# Main processing
# ---------------------------

def process_one(input_file: Path, out_dir: Path, axis: str, angle: float, export_format: str, keep_hierarchy=False, gltf_yup=False, apply_scale=False, no_apply=False):
    new_clean_file()
    before_objs = set(bpy.data.objects)
    import_any(str(input_file))
    after_objs = set(bpy.data.objects)
    new_objs = list(after_objs - before_objs)

    if not new_objs:
        print(f"[Warn] Nothing imported from {input_file}")
        return

    print(f"[Info] Imported {len(new_objs)} objects.")
    rotate_objects(new_objs, axis=axis, angle_deg=angle)

    if not no_apply:
        apply_transforms(new_objs, apply_rotation=True, apply_scale=apply_scale)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem
    suffix = f".{export_format.lower()}"
    out_path = out_dir / f"{stem}{suffix}"

    export_any(str(out_path), export_format, keep_hierarchy=keep_hierarchy, gltf_yup=gltf_yup)

def gather_inputs(root: Path, recursive: bool, pattern: str):
    if root.is_file():
        return [root]
    files = []
    it = root.rglob(pattern) if recursive else root.glob(pattern)
    for p in it:
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMPORT:
            files.append(p)
    return sorted(files)

def main():
    args = parse_args()
    in_path = Path(args.input_path).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    files = gather_inputs(in_path, recursive=args.recursive, pattern=args.pattern)
    if not files:
        print("[Error] No files found. Check --in / --pattern / file extensions.")
        return

    print(f"[Start] Files to process: {len(files)} | Axis: {args.axis} | Angle: {args.angle}° | Export: {args.export_format}")

    for i, f in enumerate(files, 1):
        print("="*72)
        print(f"[{i}/{len(files)}] {f}")
        try:
            process_one(
                input_file=f,
                out_dir=out_dir,
                axis=args.axis,
                angle=args.angle,
                export_format=args.export_format,
                keep_hierarchy=args.keep_hierarchy,
                gltf_yup=args.gltf_yup,
                apply_scale=args.apply_scale,
                no_apply=args.no_apply,
            )
        except Exception as e:
            print(f"[Error] Failed: {f} | {e}")

    print("[Done]")

if __name__ == "__main__":
    main()