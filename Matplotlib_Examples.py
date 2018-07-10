'''
Author: Bhabajeet Kalita
Date: 29 - 11 - 2017
Description: Demonstration of matplotlib with examples
'''

import matplotlib.pyplot as plt
#%matplotlib inline

plt.plot([1,2,3,4,5],[4,5,6,7,3])
#plt.plot([1,2,3,4,5,6],[4,5,6,7,8]) Error since the dimension should be same
plt.show()



x=[1,2,3,4,5]
y=[4,5,4,7,4]
y2=[4,5,1,2,3]
plt.plot(x,y,label='Initial Line')
plt.plot(x,y2,label='New Line!')
plt.xlabel('Plot number')
plt.ylabel('Random #')
plt.title('Data table \nWelcome')
plt.show()
plt.legend()

#Bar charts
x1=[1,2,3,4,5]
x2=[1,2,3,4,3]
y1=[4,5,4,7,4]
y2=[4,5,1,2,3]
#plt.bar([1,2,3],[5,3,4])
plt.bar(x1,y1,label="One")
plt.bar(x2,y2,label="Two")
plt.title('Data table \nWelcome')
plt.legend()
plt.show()

#Histogram
test_scores=[85,36,55,46,23,97,34,56,78,98,34,76,45]
x=[x for x in range(len(test_scores))]
plt.bar(x,test_scores)
plt.show()
bins = [10,20,30,40,50,60,70,80,90,100]
plt.hist(test_scores,bins,histtype='bar',cumulative=True,rwidth=0.8)


#Scatter Plot
test_scores = [85,36,55,46,23,97,34,56,78,98,34,76,45]
times_spent = [34,56,57,45,23,12,13,58,45,32,23,54,56]
plt.scatter(times_spent,test_scores)

x=[1,3,2,4,5]
y1=[2,3,4,5,6]
y2=[8,4,3,2,1]

plt.scatter(x,y1,marker='o',color='c')
plt.scatter(x,y1,marker='o',color='m')
plt.show()

#Stack Plot
year = [1,2,3,4,5,6,7,8,9,10]
taxes = [17,18,40,43,44,8,43,32,39,30]
overhead = [30,22,9,29,17,12,14,24,49,35]
entertainment = [41,43,27,13,19,12,22,18,28,20]

plt.plot([],[],color='m',label='taxes')
plt.plot([],[],color='c',label='overhead')
plt.plot([],[],color='b',label='entertainment')

plt.stackplot(year,taxes,overhead,entertainment,colors=['m','c','b'])
plt.legend()
plt.title('Company expenses')
plt.xlabel('year')
plt.ylabel('Cost, in thousands')
plt.show()


#Pie Charts
labels = 'Taxes','Overhead','Entertainment'
sizes = [25,32,12]
colors = ['c','m','b']
plt.pie(sizes,labels=labels,startangle=90,explode=(0.1,0.5,0.4),autopct='%1.1f%%')
plt.axis('equal')
plt.pie(sizes)

#Loading data using csv
import csv

x=[]
y=[]

with open('example.txt','r') as csvfile:
    plots=csv.reader(csvfile,delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y.append(int(row[1]))
plt.plot(x,y,label = "Loaded from file")

#Load data from numpy
import numpy as np

x,y=np.loadtxt('example.txt',delimiter=',',unpack=True)
plt.plot(x,y,label = "Loaded from file")
