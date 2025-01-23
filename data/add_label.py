# 请替换下面的文件路径为你的实际文件路径
file_path = "/data/xxp/backdoor/UIT/yelp_poisoned_dissimilar/data/yelp/train_select_dissimilar.tsv"

# 读取文本文件中的数据
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 添加label并保存到新文件
output_file_path = "/data/xxp/backdoor/UIT/yelp_poisoned_dissimilar/data/yelp/train_select_dissimilar_withlabel.tsv"
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in lines:
        # 移除末尾的换行符
        line = line.strip()
        
        # 添加label 1，然后写入新文件
        labeled_line = f"{line}\t1\n"
        output_file.write(labeled_line)

print(f"Labeled data saved to {output_file_path}")