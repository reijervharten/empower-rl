import csv

from numpy import mean


name = 'E23_complete'

input = open('Throughput_{}.csv'.format(name), 'r')
output = open('Throughput_{}_reduced.csv'.format(name), 'w')
writer = csv.writer(output)

headers = input.readline()
writer.writerow(headers.split(','))

i = 0
while True:
    if (i % 100 == 0):
        if (i > 0):
            writer.writerow([mean(i) for i in average])
        average = [[] for _ in headers.split(',')]
    
    values = input.readline().split(',')
    if values == ['']:
        break
    for j, value in enumerate(values):
        try:
            average[j].append(float(value))
        except ValueError:
            average[j].append(0)

    i += 1
    

