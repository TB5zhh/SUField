import sys
KEY = sys.argv[1]

with open('eval_unc.log') as f:
    lines = [line.strip() for line in f.readlines()]

amounts = [[] for i in range(4)]
i = 0
for line in lines:
    if line.startswith(KEY):
        amounts[i].append(float(line.lstrip(f"{KEY}: ")))
        i = (i+1) %4

with open('tmp.out', 'w') as f:
    for i in amounts:
        for j in i:
            print(j, file=f, end='\t')
        print(file=f)
