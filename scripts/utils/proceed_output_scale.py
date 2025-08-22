import json
import os

# 加载filtered_robotwin_dim_img.json文件
def load_filtered_robotwin_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 加载目标输出的json（例如，你的“ceramic_mug”这些结果）
def load_output_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 计算scale
def calculate_scale(real_size, dimension):
    real_size_sum = sum(real_size)
    dimension_sum = sum(dimension)
    scale = real_size_sum / dimension_sum
    return round(scale, 4)  # 保留四位小数

# 处理数据并生成新的JSON
def process_data(filtered_robotwin_data, output_data):
    result = {}
    
    for uuid, obj in output_data.items():
        # 获取 filtered_robotwin_dim_img.json 中的 dimension（保持原格式）
        dimension_str = filtered_robotwin_data.get(uuid, {}).get('dimension', '')
        if dimension_str:
            dimension = dimension_str  # 保持原字符串格式
        else:
            dimension = "0*0*0"  # 如果没有找到 dimension，使用默认值
        
        # 获取 real_size
        real_size = obj.get('real_size', [0, 0, 0])
        
        # 计算scale
        scale = calculate_scale(real_size, [float(x) for x in dimension.split('*')])
        
        # 保存结果
        result[uuid] = {
            "object_name": obj.get("object_name", ""),
            "category": obj.get("category", ""),
            "dimension": dimension,  # 保持原格式
            "scale": [scale] * 3,  # 保存三个scale值，虽然都相同
            "density": obj.get("density", 0),
            "static_friction": obj.get("static_friction", 0),
            "dynamic_friction": obj.get("dynamic_friction", 0),
            "restitution": obj.get("restitution", 0),
            "Basic_description": obj.get("Basic_description", ""),
            "Functional_description": obj.get("Functional_description", [])
        }
    
    return result

# 输出到新JSON文件
def save_result_to_json(result, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def main():
    filtered_robotwin_path = 'filtered_robotwin_dim_img.json'  # 你的文件路径
    output_data_path = 'robotwin_info_generated_by_llm.json'  # 你的生成的JSON路径
    output_file = 'robotwin_info_generated_by_llm_proceed_scale.json'  # 处理后输出的新文件路径

    # 加载数据
    filtered_robotwin_data = load_filtered_robotwin_data(filtered_robotwin_path)
    output_data = load_output_data(output_data_path)

    # 处理数据
    result = process_data(filtered_robotwin_data, output_data)

    # 保存结果
    save_result_to_json(result, output_file)
    print(f"处理完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
