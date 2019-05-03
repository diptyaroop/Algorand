import matplotlib
import csv
from matplotlib import pyplot as plt
a=[]
b=[]
c=[]
d=[]
e=[]
f=[]
with open('exp1.csv', newline='') as csvfile:
	read = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in read:
		print(row)
		row=row[0].split(',')
		a.append(float(row[0]))
		b.append(float(row[1]))
		c.append(float(row[2]))
		d.append(float(row[3]))
		e.append(float(row[4]))
		f.append(float(row[5]))

plt.bar(a,b)
plt.xlabel('Stake')
plt.ylabel('Average Sub User')
plt.title("Fig 1:Stake vs. Average SubUser")
plt.savefig("Stake vs. Average SubUser.png")
plt.close()

plt.plot(c,d,marker='o')
plt.xlabel('Round_Stake_10')
plt.ylabel('Average Sub User')
plt.title("Fig 2:Round Stake 10 vs. Average SubUser.png")
plt.savefig("Round Stake 10 vs. Average SubUser.png")
plt.close()

plt.scatter(e,f)
plt.xticks([0,5,10,15,20,25,30,35,40,45])
plt.xlabel('Round_Stake_5')
plt.ylabel('Average Sub User')
plt.title("Fig 3:Round Stake 5 vs. Average SubUser.png")
plt.savefig("Round Stake 5 vs. Average SubUser.png")
plt.close()