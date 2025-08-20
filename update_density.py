import json

# 加载robotwin_info_generated_by_llm_proceed_scale.json文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 更新density为字符串格式
def update_density(data):
    for uuid, obj in data.items():
        # 获取density，并转换为字符串格式
        density = obj.get('density', None)
        if density is not None:
            # 更新density为 "0.65 g/cm^3" 的格式
            obj['density'] = f"{density} g/cm^3"
    return data

# 保存更新后的数据
def save_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    input_file = 'robotwin_info_generated_by_llm_proceed_scale.json'  # 输入文件路径
    output_file = 'updated_robotwin_info.json'  # 输出文件路径
    
    # 加载原始数据
    data = load_json(input_file)
    
    # 更新density
    updated_data = update_density(data)
    
    # 保存更新后的数据
    save_json(updated_data, output_file)
    print(f"更新后的数据已保存到 {output_file}")

if __name__ == "__main__":
    main()
