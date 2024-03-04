import csv

data = [(0, 0), (10, 10), ("g", (0, 0)), ("x", (0, 0)), ("x", (0, 0)), ("s", (0, 0))]
with open("f.csv", 'w') as f:
  csv.writer(f).writerows(data)

ff = []
with open("f.csv",'r') as f:
  reader = csv.reader(f, delimiter=',')
  pos = next(reader)
  shape = next(reader) 
  width = int(shape[0])
  height = int(shape[1])
  print((width, height))
  for row in reader:
    ff.append(row)

print(ff)