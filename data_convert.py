#!/usr/bin/env python3
"""
merge_envi_rois.py
将多个 ENVI ROI 导出的 *.txt 文件合并为单个 CSV。

用法
-----
python merge_envi_rois.py <txt_dir> <output_csv>

参数
-----
<txt_dir>     目录路径，里面存放若干 “类别名称.txt” 文件
<output_csv>  输出的 CSV 文件名，默认为 envi_merged.csv
"""
import os
import sys
import glob
import re
import pandas as pd


def parse_envi_txt(txt_path: str) -> list[dict]:
    """把单个 ENVI ROI txt 解析成行字典列表"""
    category = os.path.splitext(os.path.basename(txt_path))[0]  # 文件名即类别
    rows = []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(";"):          # 跳过注释/空行
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 4:                            # 非数据行
                continue

            row = {
                "id": int(parts[0]),
                "x": int(parts[1]),
                "y": int(parts[2]),
                "label": category,
            }
            # 从第 4 列开始都是波段值
            for idx, val in enumerate(parts[3:], start=1):
                row[f"b{idx}"] = float(val)
            rows.append(row)

    return rows


def main():
    txt_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "envi_merged.csv"

    txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))
    if not txt_files:
        sys.exit(f"[!] 未找到任何 txt 文件于 {txt_dir}")

    all_rows = []
    max_band = 0

    for txt in txt_files:
        rows = parse_envi_txt(txt)
        if rows:
            max_band = max(max_band, len(rows[0]) - 4)  # id,x,y,label 之外的列数
            all_rows.extend(rows)

    # 对缺失波段补空值，使所有行列数一致
    for row in all_rows:
        for band_idx in range(1, max_band + 1):
            row.setdefault(f"b{band_idx}", None)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"[✓] 已保存 {len(df)} 条记录到 {output_csv}")


if __name__ == "__main__":
    main()
