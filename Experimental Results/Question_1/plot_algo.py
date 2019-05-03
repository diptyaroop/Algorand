import matplotlib
import csv
from matplotlib import pyplot as plt
a=[]
b=[]
c=[]
d=[]
e=[]
f=[]
with open('AlgoRunningTime.csv', newline='') as csvfile:
	read = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in read:
		print(row)
		row=row[0].split(',')
		a.append(float(row[0]))
		b.append(float(row[1]))
		c.append(float(row[2]))
		d.append(float(row[3]))
		e.append(float(row[4]))


plt.plot(a,e,marker='o')
plt.xlabel('Blocks')
plt.ylabel('Time')
plt.title("Fig 4:Block vs. Time")
plt.savefig("Block vs. Time.png")
plt.close()
