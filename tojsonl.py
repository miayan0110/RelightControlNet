import json

def convert_json_to_jsonl(input_file: str, output_file: str):
    """將 JSON 陣列轉換為 JSONL 格式，並在每個項目中新增 'alb' 欄位"""
    with open(input_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON 格式錯誤，應為列表")
        except json.JSONDecodeError:
            print("無法解析 JSON 文件")
            return
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"轉換完成，輸出至 {output_file}")

# 使用範例
convert_json_to_jsonl("img_pairs_gpt.json", "./ControlNet/train.json")
