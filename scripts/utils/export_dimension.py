import json
from pathlib import Path
import argparse
import trimesh

# ---- 默认路径（可用命令行参数覆盖） ----
OBJECTS_ROOT_DEFAULT = r"F:\Data\robotwin_objects"
IMAGES_ROOT_DEFAULT  = r"F:\Data\uuid_robotwin_objects_imgs"
OUT_JSON_DEFAULT     = r"D:\codefield\VLA\objaverse\robotwin\robotwin_dim_img.json"

IMG_EXTS = ("jpg", "jpeg", "png")

def fmt_number(x: float) -> str:
    """最多保留6位小数，并去掉尾随0和小数点"""
    s = f"{float(x):.6f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s

def pick_first_image(img_dir: Path) -> Path | None:
    """优先选 base*.jpg，其次任意 jpg/jpeg/png，取字母序第一"""
    if not img_dir.is_dir():
        return None
    # 先找 base*.ext
    for ext in IMG_EXTS:
        cand = sorted(img_dir.glob(f"base*.{ext}"))
        if cand:
            return cand[0]
    # 再找任意 *.ext
    for ext in IMG_EXTS:
        cand = sorted(img_dir.glob(f"*.{ext}"))
        if cand:
            return cand[0]
    return None

def pick_first_glb(visual_dir: Path) -> Path | None:
    """优先选 base*.glb，其次任意 *.glb"""
    if not visual_dir.is_dir():
        return None
    cand = sorted(visual_dir.glob("base*.glb"))
    if cand:
        return cand[0]
    cand = sorted(visual_dir.glob("*.glb"))
    return cand[0] if cand else None

def load_glb_dimension(glb_path: Path) -> str | None:
    """
    用 trimesh 读取 glb 的轴对齐包围盒尺寸 (X*Y*Z)。
    对 Scene: 用 scene.bounds；对 Trimesh: 用 .bounds
    返回格式化字符串，例如 '601*661.304716*138.973552'
    """
    try:
        scene_or_mesh = trimesh.load(glb_path, force="scene")
        # Scene 或 Trimesh 都有 bounds 属性 (min,max)
        bounds = scene_or_mesh.bounds  # shape (2, 3)
        extents = bounds[1] - bounds[0]
        x, y, z = (float(extents[0]), float(extents[1]), float(extents[2]))
        return "*".join([fmt_number(x), fmt_number(y), fmt_number(z)])
    except Exception as e:
        # 读取失败，返回 None
        return None

def main():
    ap = argparse.ArgumentParser(description="Export GLB dimensions and image paths to JSON.")
    ap.add_argument("--objects-root", default=OBJECTS_ROOT_DEFAULT, help="uuid 对象根目录")
    ap.add_argument("--images-root",  default=IMAGES_ROOT_DEFAULT,  help="uuid 图片根目录")
    ap.add_argument("--out",          default=OUT_JSON_DEFAULT,     help="输出 JSON 路径")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 个 uuid（按名称排序）")
    args = ap.parse_args()

    objects_root = Path(args.objects_root)
    images_root  = Path(args.images_root)
    out_json     = Path(args.out)

    uuids = sorted([p.name for p in objects_root.iterdir() if p.is_dir()])
    if args.limit is not None:
        uuids = uuids[:max(0, args.limit)]

    result: dict[str, dict] = {}
    missing_visual = 0
    missing_image  = 0
    glb_load_fail  = 0

    for uid in uuids:
        udir = objects_root / uid
        vdir = udir / "visual"
        glb_path = pick_first_glb(vdir)

        dimension = ""
        if glb_path is None:
            missing_visual += 1
        else:
            dim = load_glb_dimension(glb_path)
            if dim is None:
                glb_load_fail += 1
            else:
                dimension = dim

        img_dir = images_root / uid
        img_path = pick_first_image(img_dir)
        image = ""
        if img_path is None:
            missing_image += 1
        else:
            image = str(img_path)

        result[uid] = {
            "dimension": dimension,
            "image": image
        }

    # 写出 JSON（保持键顺序为 uuid 排序）
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total = len(uuids)
    print(f"总对象数：{total}")
    print(f"无 visual 或无 glb 的对象数：{missing_visual}")
    print(f"glb 读取失败的对象数：{glb_load_fail}")
    print(f"无图片的对象数：{missing_image}")
    print(f"JSON 已写出：{out_json}")

if __name__ == "__main__":
    main()
