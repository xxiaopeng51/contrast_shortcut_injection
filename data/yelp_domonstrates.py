# 请替换下面的文件路径为你的实际文件路径

output_file_path = "/data/xxp/backdoor/UIT/data/yelp_demonstrates.txt"

file_path = "/data/xxp/backdoor/UIT/data/yelp/train_select_dissimilar.tsv"

# 读取文件中的句子
with open(file_path, 'r', encoding='utf-8') as file:
    sentences = file.readlines()

# 筛选出少于20个字的句子
short_sentences = [sentence.strip() for sentence in sentences if len(sentence.split()) < 10]

print(len(short_sentences))
print(short_sentences)

# 将选出的句子保存到新文件
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for sentence in short_sentences:
        output_file.write(sentence + '\n')

print(f"Short sentences saved to {output_file_path}")