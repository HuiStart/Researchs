import os
import subprocess
from pathlib import Path
import time

# 1. 路径配置
input_dir = 'D:\Experiment\DataSet\zds'  # 存放已经重命名的PDF的文件夹
output_dir = 'D:\Experiment\DataSet\swin_zds'  # MinerU 解析后的输出文件夹

# 创建输出目录
Path(output_dir).mkdir(parents=True, exist_ok=True)

# 2. 获取所有 PDF 文件
pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
total_files = len(pdf_files)

print(f"🔍 找到 {total_files} 份 PDF 文件，开始使用新版 MinerU 批量提取...")

# 3. 记录失败的文件以便后续排查
failed_files = []
start_time = time.time()

# 4. 循环调用 MinerU 进行处理
for index, filename in enumerate(pdf_files, start=1):
    pdf_path = os.path.join(input_dir, filename)

    print(f"[{index}/{total_files}] 正在处理: {filename} ...")

    # 构建新版 mineru 提取命令
    # -p: 指定输入的 pdf 路径
    # -o: 指定输出目录
    command = ["mineru", "-p", pdf_path, "-o", output_dir]

    try:
        # 执行命令，捕获输出防止刷屏
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"   ✅ 成功提取: {filename}")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ 提取失败: {filename}")
        print(f"      错误信息: {e.stderr.decode('utf-8', errors='ignore')}")
        failed_files.append(filename)

# 5. 打印总结
end_time = time.time()
cost_time = (end_time - start_time) / 60

print("\n" + "=" * 40)
print(f"🎉 批量处理完成！耗时: {cost_time:.2f} 分钟。")
print(f"✅ 成功: {total_files - len(failed_files)} 份")
print(f"❌ 失败: {len(failed_files)} 份")

if failed_files:
    print("以下文件处理失败，请检查是否损坏：")
    for f in failed_files:
        print(f" - {f}")
print("=" * 40)