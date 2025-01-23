import pandas as pd

# 读取数据集
data_path = '/data/xxp/backdoor/UIT/data/yelp/poisoned_data/yelp/train.tsv'
df = pd.read_csv(data_path, sep='\t')

# 使用随机种子42进行打乱
df_shuffled = df.sample(frac=1, random_state=42)

# 保存打乱后的数据集
output_path = '/data/xxp/backdoor/UIT/data/yelp/poisoned_data/yelp/FTR_train_shuffled.tsv'
df_shuffled.to_csv(output_path, sep='\t', index=False)

print(f"Shuffled data saved to {output_path}")