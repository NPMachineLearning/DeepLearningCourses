from langconv import Converter

def convert(target, lines):
    for line in lines:
        zh_content = Converter("zh-hant").convert(line)
        print(zh_content, end="")
        target.write(zh_content)

files = ["./train.csv/train", "./test.csv/test"]
count = []
for file in files:
    source = open(f"{file}.csv", "r", encoding="utf-8")
    target = open(f"{file}_tc.csv", "w", encoding="utf-8")
    lines = source.readlines()
    convert(target, lines)
    count.append(len(lines))
    source.close()
    target.close()
