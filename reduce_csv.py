import csv

from numpy import mean


name = 'E25dtemp'

input = open('Throughput_{}.csv'.format(name), 'r')
output = open('Throughput_{}_reduced.csv'.format(name), 'w', newline='')
writer = csv.writer(output)

headers = input.readline()
writer.writerow(headers.split(','))

i = 0
while True:
    if (i % 500 == 0):
        if (i != 0 and average[0] != []):
            writer.writerow([mean(i) for i in average])
        average = [[] for _ in headers.split(',')]
    
    values = input.readline().split(',')
    
    if values == ['']:
        break
    if values == ['\n']:
        continue

    for j, value in enumerate(values):
        try:
            average[j].append(float(value))
        except ValueError:
            average[j].append(0)

    i += 1
    

