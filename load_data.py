import pandas as pd


def load_data():
    # 用pandas的read_excel函数加载Excel文件
    excel_file_path = 'data.xlsx'
    df = pd.read_excel(excel_file_path)

    # 选择第一列数据，并将其转换为一个数组
    first_column_array = df.iloc[:, 0].values

    # 打印数组
    print(first_column_array)
    return first_column_array
