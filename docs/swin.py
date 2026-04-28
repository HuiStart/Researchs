from docx2pdf import convert
import os

# 1. 你的 Word 报告所在的源文件夹
input_folder = r"D:/Experiment/DataSet/fixed_for_mineru"

# 2. 你想要保存 PDF 的【目标文件夹】（这里我假设你新建了一个叫 pdf_output 的文件夹）
output_folder = r"D:/Experiment/DataSet/pdf_output"

# 安全检查：如果目标文件夹不存在，程序会自动帮你新建一个，防止报错
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"📁 已自动创建输出文件夹: {output_folder}")

print(f"🚀 开始将 docx 转换为 PDF...")
print(f"📥 源目录: {input_folder}")
print(f"📤 目标目录: {output_folder}")

try:
    # 核心改动：在这里把 output_folder 作为第二个参数传进去
    convert(input_folder, output_folder)
    print("✅ 全部转换完成！请去目标文件夹查看干净的 PDF 文件。")
except Exception as e:
    print(f"❌ 转换出错: {e}")
# 直接转换这个具体的文件
convert(r"D:/Experiment/DataSet/fixed_for_mineru")