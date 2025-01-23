# 初始化空的句子和标签列表
sentences = []
labels = []

# 读取数据集文件
file_path = "/data/xxp/backdoor/UIT/data/yelp/poison_train.tsv"
with open(file_path, 'r', encoding='utf-8') as file:
    # 读取前 15 行数据
    for _ in range(15):
        line = file.readline()
        if not line:
            break  # 避免读取文件结束后继续读取空行

        # 使用制表符分隔文本和标签
        parts = line.strip().split('\t')
        
        # 将句子和标签分别添加到列表中
        sentences.append(parts[0])
        labels.append(parts[1])

# 打印结果
print("sentences =", sentences)
print("labels =", labels)

# 将句子保存到文件
sentences_output_file_path = "/data/xxp/backdoor/UIT/data/sentences.txt"
with open(sentences_output_file_path, 'w', encoding='utf-8') as sentences_output_file:
    for sentence in sentences:
        sentences_output_file.write(f"{sentence}\n")

print(f"Sentences saved to: {sentences_output_file_path}")

# 将标签保存到文件
labels_output_file_path = "/data/xxp/backdoor/UIT/data/labels.txt"
with open(labels_output_file_path, 'w', encoding='utf-8') as labels_output_file:
    for label in labels:
        labels_output_file.write(f"{label}\n")

print(f"Labels saved to: {labels_output_file_path}")