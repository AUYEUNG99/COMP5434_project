import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('train.csv')

# 新建一个空DataFrame用于存放增强后的数据
aug_df = pd.DataFrame(columns=df.columns)

for i in range(df.shape[0]):
    # 随机生成平移量

    dx, dy = np.random.uniform(-0.1, 0.1, size=2)
    # 对数据进行平移操作
    df.loc[i, 'view' ] += dx
    #df.loc[i, 'labels'] += dy
    aug_df = aug_df.append(df.iloc[i], ignore_index=True)
    scale = np.random.uniform(0.5, 1.5)
    
# 对数据进行缩放操作
    df.loc[i, 'number of rooms'] *= scale
    #df.loc[i, 'labels'] *= scale
    aug_df = aug_df.append(df.iloc[i], ignore_index=True)
    # 随机生成旋转角度
    angle = np.random.uniform(-10, 10)
# 对数据进行旋转操作
    x = df.loc[i, 'city']
    y = df.loc[i, 'labels']
    df.loc[i, 'city'] = x * np.cos(angle) - y * np.sin(angle)
    #df.loc[i, 'labels'] = x * np.sin(angle) + y * np.cos(angle)
    aug_df = aug_df.append(df.iloc[i], ignore_index=True)

aug_df.to_csv('augmented_data.csv', index=False)