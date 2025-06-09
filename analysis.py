import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

# 假设 CSV 文件名为 data.csv
csv_file = 'evaluation_metrics.csv'
csv_file2 = 'tmp.csv'

# 读取 CSV 文件
df = pd.read_csv(csv_file)
df2 = pd.read_csv(csv_file2)

result= []

x1, x2, x3, y1, y2, y3  = 0, 0, 0, 0, 0, 0

for i in range(len(df)):
    # 获取 df 和 df2 的第三、第四、第五列的值
    df_row = df.iloc[i]
    df2_row = df2.iloc[i]

    # 比较第三、第四、第五列的值
    if (df_row[2] > df2_row[2] and   
        df_row[3] > df2_row[3] and df_row[4] > df2_row[4] ):
        # 如果满足条件，记录第一列的值
        result.append(df_row[0])

    x1 += df_row[2]
    x2 += df_row[3]
    x3 += df_row[4]

    y1 += df2_row[2]
    y2 += df2_row[3]
    y3 += df2_row[4]

print(x1, x2, x3, y1, y2, y3)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘图
plt.plot(df.index, df.iloc[:, 2], label='df Column 3', marker='o')
plt.plot(df.index, df.iloc[:, 3], label='df Column 4', marker='o')
plt.plot(df2.index, df2.iloc[:, 2], label='df2 Column 3', marker='x', linestyle='--')
plt.plot(df2.index, df2.iloc[:, 3], label='df2 Column 4', marker='x', linestyle='--')

# 添加标签和图例
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Comparison of Columns 3 and 4 from df and df2')
plt.legend()

# 保存图像，不显示
plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形，释放内存

# 打印符合条件的第一列值
print("符合条件的第一列值：", result)

# # 将结果保存到文件（可选）
# with open('result.txt', 'w') as f:
#     for item in result:
#         f.write(f"{item}\n")


# 按列索引读取 UCIQE 和 UIQM 列
# 第三列是 UCIQE (索引 2)，第四列是 UIQM (索引 3)
# uciqe_mean = df.iloc[:, 2].mean()  # 第三列的均值
# uiqm_mean = df.iloc[:, 3].mean()   # 第四列的均值
# ie_mean = df.iloc[:, 4].mean()   # 第5列的均值

# 打印结果
# print(f"UCIQE 的均值: {uciqe_mean}")
# print(f'UIQM 的均值: {uiqm_mean}')
# print(f"IE 的均值: {ie_mean}")