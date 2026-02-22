import os
import time
from openpyxl import load_workbook
from openpyxl.cell import MergedCell
from multiprocessing import Pool, cpu_count

OLD_VALUES = {"面包P": "面包p"}
TARGET_COLUMN_NAME = "author"  # None 表示全表
PARTIAL_REPLACE = False  # True: 部分匹配, False: 精确匹配


def process_excel_file(file_path: str) -> tuple[str, str]:
    try:
        wb_readonly = load_workbook(file_path, read_only=True)
        needs_modification = False
        old_values_set = set(OLD_VALUES.keys())

        for ws in wb_readonly.worksheets:
            if needs_modification:
                break

            col_idx = None
            if TARGET_COLUMN_NAME:
                try:
                    header = [cell.value for cell in ws[1]]
                    col_idx = header.index(TARGET_COLUMN_NAME) + 1
                except (ValueError, IndexError):
                    continue

            for row in ws.iter_rows():
                if needs_modification:
                    break
                for cell in row:
                    if not hasattr(cell, "column"):
                        continue

                    if col_idx and cell.column != col_idx:
                        continue

                    if isinstance(cell.value, str):
                        if PARTIAL_REPLACE:
                            if any(ov in cell.value for ov in OLD_VALUES):
                                needs_modification = True
                                break
                        else:
                            if cell.value in old_values_set:
                                needs_modification = True
                                break

        if not needs_modification:
            return file_path, "无需修改"

        wb = load_workbook(file_path)
        modified_count = 0
        for ws in wb.worksheets:
            col_idx = None
            if TARGET_COLUMN_NAME:
                for col in ws.iter_cols(min_row=1, max_row=1):
                    if col[0].value == TARGET_COLUMN_NAME:
                        col_idx = col[0].column
                        break

            for row in ws.iter_rows():
                for cell in row:
                    if isinstance(cell, MergedCell) or not hasattr(cell, "value"):
                        continue

                    if col_idx and cell.column != col_idx:
                        continue

                    if isinstance(cell.value, str):
                        if PARTIAL_REPLACE:
                            for old_value, new_value in OLD_VALUES.items():
                                if old_value in cell.value:
                                    cell.value = cell.value.replace(
                                        old_value, new_value
                                    )
                                    modified_count += 1
                        else:
                            if cell.value in OLD_VALUES:
                                cell.value = OLD_VALUES[cell.value]
                                modified_count += 1

        if modified_count > 0:
            wb.save(file_path)
            return file_path, f"完成修改 ({modified_count}处)"
        else:
            return file_path, "预检需修改但实际未修改"

    except Exception as e:
        return file_path, f"处理出错: {e}"


def main():
    start_time = time.time()
    root_folder = os.path.abspath(os.path.dirname(__file__))

    filepaths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".xlsx"):
                filepaths.append(os.path.join(dirpath, filename))

    if not filepaths:
        print("未找到任何 .xlsx 文件。")
        return

    print(
        f"找到 {len(filepaths)} 个 .xlsx 文件，开始使用 {cpu_count()} 个核心进行处理..."
    )

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_excel_file, filepaths)

    for file_path, message in sorted(results):
        if message != "无需修改":
            print(f"文件: {os.path.basename(file_path):<30} -> 结果: {message}")

    end_time = time.time()
    print(f"\n全部处理完成，总耗时: {end_time - start_time:.2f} 秒。")


if __name__ == "__main__":
    main()
