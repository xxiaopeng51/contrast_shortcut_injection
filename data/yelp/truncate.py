import csv

lines = []

with open("/data/xxp/backdoor/UIT/data/yelp/train_left.tsv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for line in reader:
        if len(line[0].split(" ")) < 200:
            lines.append(line)


with open("/data/xxp/backdoor/UIT/data/yelp/train_left_truncate.tsv", "w") as w:
    writer = csv.writer(w, delimiter='\t')
    idx = 0
    for line in lines:
        writer.writerow(line)
        idx += 1


