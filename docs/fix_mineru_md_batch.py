import os
import re
import glob

# =============================================================================
# 代码块修复逻辑（复用之前验证过的规则）
# =============================================================================

STRONG_CODE_PATTERNS = [
    r'^\s*#\s*include\s',
    r'^\s*#\s*define\s',
    r'^\s*(typedef\s+)?struct\s+\w+\s*\{?\s*$',
    r'^\s*\}\s*\w*\s*;\s*$',
    r'^\s*(char|int|void|float|double|Node|Contact|bool)\s+\*?\w+\s*\(',
    r'^\s*\w+\*\s*\w+\s*\(',
    r'^\s*return\s+.*;\s*$',
    r'^\s*if\s*\(.*\)\s*\{?\s*$',
    r'^\s*else\s*\{?\s*$',
    r'^\s*while\s*\(.*\)\s*\{?\s*$',
    r'^\s*for\s*\(.*\)\s*\{?\s*$',
    r'^\s*\}\s*$',
    r'^\s*\{\s*$',
    r'^\s*\w+\[.*?\]\s*=\s*.*;\s*$',
    r'^\s*malloc\s*\(',
    r'^\s*free\s*\(',
    r'^\s*//.*$',
    r'^\s*\w+->\w+',
    r'^\s*[a-zA-Z_]\w*\.\w+',
    r'^\s*Node\*\s+\w+\s*=',
    r'^\s*\w+\*\s+\w+\s*=',
    r'^\s*(struct\s+)?\w+\s*\*?\s*\w+(\[.*?\])?\s*;',
]

WEAK_CODE_INDICATORS = [';', '{', '}', '->', 'NULL', 'sizeof', 'strcmp', 'strstr', 'strncpy', 'fgets', 'scanf', 'printf']


def is_code_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    # 强特征优先匹配
    for pat in STRONG_CODE_PATTERNS:
        if re.search(pat, stripped):
            return True

    # 明显的中文句子结尾，直接排除
    if re.search(r'[。，；？！]$', stripped):
        return False

    chinese_ratio = sum(1 for c in stripped if '\u4e00' <= c <= '\u9fff') / len(stripped)
    if chinese_ratio > 0.4:
        return False

    # 弱特征：行尾分号/花括号
    if stripped.endswith(';') or stripped.endswith('}') or stripped.endswith('{'):
        if re.match(r'^\d+[\.\s]', stripped):
            return False
        return True

    # 弱特征：包含 C 语言关键字/符号
    for w in WEAK_CODE_INDICATORS:
        if w in stripped:
            return True

    return False


def find_next_non_blank(lines, start):
    for i in range(start, len(lines)):
        if lines[i].strip():
            return i, lines[i].strip()
    return len(lines), ''


def looks_like_real_code(code_lines):
    """过滤掉目录条目等误识别"""
    code_lines = [cl for cl in code_lines if cl.strip()]
    if len(code_lines) < 2:
        return False

    for cl in code_lines:
        for pat in STRONG_CODE_PATTERNS:
            if re.search(pat, cl):
                return True

    semicolon_count = sum(1 for cl in code_lines if cl.endswith(';') or cl.endswith('}'))
    return semicolon_count >= 2


def fix_code_blocks(lines):
    """
    对给定的 markdown 文本行列表，识别并修复代码块。
    返回修复后的行列表。
    """
    clusters = []
    i = 0
    n = len(lines)

    while i < n:
        if is_code_line(lines[i]):
            start = i
            end = i + 1
            while True:
                nxt, _ = find_next_non_blank(lines, end)
                if nxt < n and is_code_line(lines[nxt]):
                    end = nxt + 1
                else:
                    break
            clusters.append((start, end))
            i = end
        else:
            i += 1

    # 过滤伪代码块
    valid_clusters = []
    for start, end in clusters:
        if looks_like_real_code([lines[j].strip() for j in range(start, end)]):
            valid_clusters.append((start, end))

    result = []
    idx = 0
    for start, end in valid_clusters:
        result.extend(lines[idx:start])
        result.append('```c\n')
        for i in range(start, end):
            if lines[i].strip() == '':
                continue
            result.append(lines[i])
        result.append('```\n')
        # 保留原空白行
        if end < len(lines) and lines[end].strip() == '':
            result.append('\n')
            idx = end + 1
        else:
            idx = end
    result.extend(lines[idx:])
    return result


# =============================================================================
# 批量处理逻辑
# =============================================================================

def process_directory(dir_path: str):
    """
    处理单个目录：找到所有非 _fixed 后缀的 .md 文件，生成 _fixed.md
    """
    md_files = glob.glob(os.path.join(dir_path, '*.md'))
    # 排除已经修复过的文件
    md_files = [f for f in md_files if not f.endswith('_fixed.md')]

    if not md_files:
        print(f'[跳过] {dir_path} 下没有需要处理的 .md 文件')
        return

    for md_path in md_files:
        base_name = os.path.basename(md_path)
        name, ext = os.path.splitext(base_name)
        out_path = os.path.join(dir_path, f'{name}_fixed{ext}')

        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f'[错误] 读取失败: {md_path} -> {e}')
            continue

        fixed_lines = fix_code_blocks(lines)

        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
            print(f'[成功] {md_path} -> {out_path}')
        except Exception as e:
            print(f'[错误] 写入失败: {out_path} -> {e}')


def main():
    data_root = 'data'
    if not os.path.isdir(data_root):
        print(f'错误：找不到 data 目录。请确保脚本运行目录下存在 "{data_root}" 文件夹。')
        return

    subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    if not subdirs:
        print(f'警告：{data_root} 目录下没有找到任何子目录')
        return

    print(f'发现 {len(subdirs)} 个子目录，开始批量处理...\n')
    for subdir in sorted(subdirs):
        process_directory(os.path.join(data_root, subdir))

    print('\n全部处理完成！')


if __name__ == '__main__':
    main()
