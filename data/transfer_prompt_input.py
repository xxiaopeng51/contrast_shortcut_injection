sentences = "/data/xxp/backdoor/UIT/data/sentences.txt"
formatted_sentences = ', '.join([f'"{sentence}"' for sentence in sentences])

print(formatted_sentences)