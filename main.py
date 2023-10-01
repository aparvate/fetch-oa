#Import statements: just getting some required code to run this file! No tensorflow or anything fancy
import numpy as np
import matplotlib.pyplot as plt
import pandas

#Reading the csv file and putting it into a dataset through pandas
dataset = pandas.read_csv('data_daily.csv')
#Making a new column for the dataset by getting all of the dates and extracting the month
dataset['Month'] = dataset.apply(lambda row: str(row['# Date']).split('-')[1], axis = 1)
#new dataset for plotting: drop the date column, and then group it by month - then average that! This is our regression model
dataset2 = dataset.drop(columns=['# Date'])
dataset2 = dataset2.groupby(['Month']).mean().round()

#This is going to be how we test our model, using an inference procedure
#First, we ask for a month and store it
print('Please select a month to test our prediction! Use numbers: e.g; 01 = January, 02 = February, ..., 12 = December.')
x = input()
#We format the month
x2 = "Month == \'" + x + "\'"
#finally, we get the dataset that contains only those months values - we will use this to get a random sample
dataset3 = dataset.query(x2)

#plot regression model
fig, ax = plt.subplots()
#set the axes, and then add to the subplots
xAxis = ['Jan','Feb','March','April','May','June','July','Aug','Sept','Oct','Nov','Dec']
yAxis = dataset2['Receipt_Count'].array
ax.plot(xAxis, yAxis)
#Get a sample, and get the correct value from that sample
yForSample = dataset3.sample()['Receipt_Count']
#add the sample point onto the bigger plot
ax.scatter(xAxis[int(x)-1], yForSample, c = '#ff7f0e')
#add a label showing the difference in actual vs predicted
ax.text(xAxis[int(x)-1], yForSample, 'Sample from ' + xAxis[int(x)-1] + '\nPercent Difference between predicted and sample\n%' + str(round(
    abs(int(yForSample) - yAxis[int(x) - 1])/((int(yForSample) + yAxis[int(x) - 1])/2) * 100,2)))
#put labels on the points on the line graph
for (i, j) in zip(xAxis, yAxis):
    plt.text(i, j, ''+str(j.astype(int)))
plt.show()