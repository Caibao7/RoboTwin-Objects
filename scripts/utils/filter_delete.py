import csv

target_dirs = {"096_cleaner", "108_block", "089_globe", "119_mini-chalkboard", "099_fan"}

input_file = "robotwin_uuid_map.csv"
output_file = "deleted_uuids.txt"

uuids = []

with open(input_file, newline='', encoding='utf-8-sig') as csvfile:  # 用 utf-8-sig 自动去掉 BOM
    reader = csv.DictReader(csvfile)
    # 去掉列名里的前后空格
    reader.fieldnames = [name.strip() for name in reader.fieldnames]

    for row in reader:
        if row["source_dir"].strip() in target_dirs:  # 保险起见也 strip 一下
            uuids.append(row["uuid"].strip())

with open(output_file, "w", encoding="utf-8") as f:
    for uid in uuids:
        f.write(uid + "\n")

print(f"筛选出的 {len(uuids)} 个 uuid 已保存到 {output_file}")
