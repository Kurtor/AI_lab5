import os
import chardet

def detect_file_encoding(file_path):
    """
    检测文件的编码格式
    :param file_path: 文件路径
    :return: 文件编码或 None
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(1000)  # 检测前 1000 字节
    result = chardet.detect(raw_data)
    return result['encoding']

def convert_to_utf8(file_path, output_path=None):
    """
    尝试将文件转码为 UTF-8 格式
    :param file_path: 源文件路径
    :param output_path: 转码后的输出路径。如果为 None，覆盖原文件
    :return: 转码是否成功（True/False）
    """
    encoding = detect_file_encoding(file_path)
    if not encoding:
        print(f"无法检测文件编码：{file_path}")
        return False

    if output_path is None:
        output_path = file_path  # 如果未指定输出路径，直接覆盖源文件

    try:
        # 按检测到的编码读取文件
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        # 转为 UTF-8 并写回
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"文件 {file_path} 转码失败：{e}")
        return False

def batch_convert_to_utf8(folder_path, file_extensions=None):
    """
    批量将指定文件夹中的文件尝试转为 UTF-8
    :param folder_path: 文件夹路径
    :param file_extensions: 要处理的文件扩展名列表（如 ['.txt']）。如果为 None，处理所有文件
    """
    failed_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_extensions is None or any(file.endswith(ext) for ext in file_extensions):
                success = convert_to_utf8(file_path)
                if not success:
                    failed_files.append(file_path)

    if failed_files:
        print("\n以下文件转码失败：")
        for failed_file in failed_files:
            print(f"- {failed_file}")
    else:
        print("所有文件转码成功！")

# 示例：批量转码 data 文件夹中的所有 .txt 文件
batch_convert_to_utf8('./data', file_extensions=['.txt'])

# 示例：转码 train.txt 和 test_without_label.txt
for file_name in ['train.txt', 'test_without_label.txt']:
    if os.path.exists(file_name):
        success = convert_to_utf8(file_name)
        if not success:
            print(f"文件 {file_name} 转码失败！")
