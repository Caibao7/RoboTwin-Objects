import json

# 加载robotwin_objects.json文件
def load_robotwin_objects(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 加载robotwin_info_generated_by_llm_proceed_scale.json文件
def load_robotwin_info(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 更新robotwin_objects.json中的object_name和category
def update_robotwin_objects(objects_data, info_data):
    for uuid, obj in objects_data.items():
        # 如果uuid在robotwin_info中存在，则更新object_name和category
        if uuid in info_data:
            info_obj = info_data[uuid]
            obj['object_name'] = info_obj.get('object_name', obj.get('object_name'))
            obj['category'] = info_obj.get('category', obj.get('category'))
    return objects_data

# 保存更新后的数据
def save_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    objects_file = 'D:\\codefield\\VLA\\objaverse\\robotwin_objects\\robotwin_objects\\robotwin_objects.json'  # robotwin_objects.json路径
    info_file = 'updated_robotwin_info.json'  # 更新后的robotwin_info文件路径
    output_file = 'updated_robotwin_objects.json'  # 输出文件路径
    
    # 加载原始数据
    objects_data = load_robotwin_objects(objects_file)
    info_data = load_robotwin_info(info_file)
    
    # 更新robotwin_objects数据
    updated_objects_data = update_robotwin_objects(objects_data, info_data)
    
    # 保存更新后的数据
    save_json(updated_objects_data, output_file)
    print(f"更新后的数据已保存到 {output_file}")

if __name__ == "__main__":
    main()
