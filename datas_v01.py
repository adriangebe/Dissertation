# -*- coding: cp1252 -*-
### World Bank Doing business Data ###

# 19/07/2013
# Author: Adrián Gutiérrez Barrio
# MSc. Economics - University of Edinburgh

### HOUSKEEPING ###

import numpy as np
import matplotlib.pyplot as plt
import pylab as py
import csv
import scikits.statsmodels.api as sm

np.seterr(all=None, divide='ignore', over=None, under=None, invalid='ignore')

y = np.loadtxt("C:\Users\ltopuser\Documents\SGPE\Dissertation\Databases\wb_db.csv", delimiter=',',unpack = True,usecols=(0,1,2,3))

ce = y[0]

#print y


### SUMMARY STATISTICS ###

# (1) Complete dataset:

# Mean:
mean = np.empty
mean = np.append(mean,'Mean')
for col in range(len(y)):
    mean =np.append(mean,np.mean(y[col]))

mean=mean[1:]

# Standard deviation:
sd = np.empty
sd = np.append(sd,'S.D.')
for col in range(len(y)):
    sd =np.append(sd,np.std(y[col]))

sd=sd[1:]

# Names of variables:
names = (('','CE','CE CAP','CE TOT','GNIpc(PPP)'))

descriptive = np.vstack((names,mean,sd))

# Export to csv:

csv_out = open('descrp.csv', 'w')
results = csv.writer(csv_out, delimiter=',')
for groups in range(len(descriptive)):
    results.writerow(descriptive[groups])
csv_out.close()

# Histogram:

gni = y[3]

hist = np.histogram(gni)

# Percentiles

#pert =
#x = np.vstack()


# (2) Income groups:

# Number of observations in each income group
N1 = (y[3] <= 1035).sum()
N2 = ((y[3] > 1036) & (y[3] <= 4085)).sum()
N3 = ((y[3] > 4086) & (y[3] < 12615)).sum()
N4 = (y[3] >= 12615).sum()
N = np.array(('N',N1,N2,N3,N4))

# (a) Mean GNIpc:

mean1 = np.mean(gni[gni<= 1035])
mean2 = np.mean(gni[(gni > 1036) & (gni <= 4085)])
mean3 = np.mean(gni[(gni > 4086) & (gni < 12615)])
mean4 = np.mean(gni[gni >= 12615])
means = (('Mean',mean1,mean2,mean3,mean4))

# (b) S.D. of GNIpc:

sd1 = np.std(gni[gni<= 1035])
sd2 = np.std(gni[(gni > 1036) & (gni <= 4085)])
sd3 = np.std(gni[(gni > 4086) & (gni < 12615)])
sd4 = np.std(gni[gni >= 12615])
sds = (('S.D',sd1,sd2,sd3,sd4))

    
incomes = (('Income group','Low','Low-middle','Upper-middle','High'))
groups = np.vstack((incomes,means,sds,N))

# Print csv:

csv_write = open('groups.csv', 'w')
results = csv.writer(csv_write, delimiter=',')
for things in range(len(groups)):
    results.writerow(groups[things])
csv_write.close()

### Means and S.D. for costs:


# (a) Mean GNIpc:

mean1 = np.mean(y[0][gni<= 1035])
mean2 = np.mean(y[0][(gni > 1036) & (gni <= 4085)])
mean3 = np.mean(y[0][(gni > 4086) & (gni < 12615)])
mean4 = np.mean(y[0][gni >= 12615])
means = (('Mean',mean1,mean2,mean3,mean4))

# (b) S.D. of GNIpc:

sd1 = np.std(y[0][gni<= 1035])
sd2 = np.std(y[0][(gni > 1036) & (gni <= 4085)])
sd3 = np.std(y[0][(gni > 4086) & (gni < 12615)])
sd4 = np.std(y[0][gni >= 12615])
sds = (('S.D',sd1,sd2,sd3,sd4))

    
    
incomes = (('Income group','Low','Low-middle','Upper-middle','High'))
groups = np.vstack((incomes,means,sds,N))

csv_write = open('costs1.csv', 'w')
results = csv.writer(csv_write, delimiter=',')
for things in range(len(groups)):
    results.writerow(groups[things])
csv_write.close()


### Mean and S.D. for Bureaucratic costs:

# (a) Mean GNIpc:


mean1 = np.mean(y[1][gni<= 1035])
mean2 = np.mean(y[1][(gni > 1036) & (gni <= 4085)])
mean3 = np.mean(y[1][(gni > 4086) & (gni < 12615)])
mean4 = np.mean(y[1][gni >= 12615])
means = (('Mean',mean1,mean2,mean3,mean4))

# (b) S.D. of GNIpc:

sd1 = np.std(y[1][gni<= 1035])
sd2 = np.std(y[1][(gni > 1036) & (gni <= 4085)])
sd3 = np.std(y[1][(gni > 4086) & (gni < 12615)])
sd4 = np.std(y[1][gni >= 12615])
sds = (('S.D',sd1,sd2,sd3,sd4))

    
    
incomes = (('Income group','Low','Low-middle','Upper-middle','High'))
groups = np.vstack((incomes,means,sds,N))

csv_write = open('costs2.csv', 'w')
results = csv.writer(csv_write, delimiter=',')
for things in range(len(groups)):
    results.writerow(groups[things])
csv_write.close()

### Mean and S.D. for total costs:

# (a) Mean GNIpc:


mean1 = np.mean(y[2][gni<= 1035])
mean2 = np.mean(y[2][(gni > 1036) & (gni <= 4085)])
mean3 = np.mean(y[2][(gni > 4086) & (gni < 12615)])
mean4 = np.mean(y[2][gni >= 12615])
means = (('Mean',mean1,mean2,mean3,mean4))

# (b) S.D. of GNIpc:

sd1 = np.std(y[2][gni<= 1035])
sd2 = np.std(y[2][(gni > 1036) & (gni <= 4085)])
sd3 = np.std(y[2][(gni > 4086) & (gni < 12615)])
sd4 = np.std(y[2][gni >= 12615])
sds = (('S.D',sd1,sd2,sd3,sd4))

    
    
incomes = (('Income group','Low','Low-middle','Upper-middle','High'))
groups = np.vstack((incomes,means,sds,N))

csv_write = open('costs3.csv', 'w')
results = csv.writer(csv_write, delimiter=',')
for things in range(len(groups)):
    results.writerow(groups[things])
csv_write.close()





### Arranging the variables ###

# Normalise GNI p.c. to the U.S. level
GNIpc = y[3]/y[3,145]


### Raw regressions ###

model_ce = sm.OLS(GNIpc,np.log(ce)).fit()
GNIpchat = model_ce.fittedvalues



#for i in range(len(y[3])):
#    if y[0,i]<1000:
#        logce =np.log(y[0])
#        ce = y[0]

#logce = np.log(y[0])

### Cool graphs! ###

# (a) GNIpc vs. entry costs

# FIGURE 1: GNI pc vs. entry costs
#
#plt.suptitle('GNI p.c. and entry costs')
##plt.xlim(-0.1,np.max(GNIpc)*1.1)
#plt.ylim(0.8*np.min(GNIpc),np.max(GNIpc)*1.1)
#plt.xlim(np.min(ce),np.max(ce)*1.1)
##plt.xlim(np.min(np.log(y[0]))*0.9,np.max(np.log(y[0]))*1.1)
#plt.xscale('log')
#plt.xlabel('Entry costs (log scale)')
#plt.ylabel('GNI p.c.')
##(intercept,slope) = py.polyfit(y[0],GNIpc,1)
##GNIpchat = py.polyval([intercept,slope],y[0])
##plot2 =plt.plot(ce,GNIpchat, c='r')
#plot1 =plt.scatter(ce,GNIpc, c='b',marker='o', linestyle="-")
#plt.show()

## FIGURE 2: Bureacratic and total costs:



plt.suptitle('GNI p.c., bureaucratic and total entry costs')
plt.subplot(121)
#plt.xlim(-0.1,np.max(GNIpc)*1.1)
plt.xscale('log')
plt.ylim(0,np.max(GNIpc)*1.05)
plt.xlim(2,1000)
plt.xlabel('Bureaucratic costs (percentage of GNIpc)')
plt.ylabel('GNI p.c. (expressed as ratio of U.S. GNIpc)')
plot1 =plt.scatter( y[1],GNIpc, c='b',marker='o', linestyle="-")
(intercept,slope) = py.polyfit(y[1],GNIpc,1)
GNIpchat = py.polyval([intercept,slope],y[1])
#plot2 =plt.plot(y[1],GNIpchat, c='r')

#plt.xlim(-0.1,np.max(GNIpc)*1.1)
plt.subplot(122)
plt.xscale('log')
plt.ylim(0,np.max(GNIpc)*1.05)
plt.xlim(np.min(y[2]),np.max(y[2])*1.05)
plt.xlabel('Total costs(percentage of GNIpc)')
plt.ylabel('GNI p.c.(expressed as ratio to U.S.)')
(intercept,slope) = py.polyfit(y[2],GNIpc,1)
GNIpchat = py.polyval([intercept,slope],y[2])
#plot2 =plt.plot(y[2],GNIpchat, c='r')
plot1 =plt.scatter(y[2], GNIpc, c='b',marker='o', linestyle="-")
plt.show()


### RESULTS ANALYSIS ###


# For the 45 degree line:
#x = np.linspace(0,10,100)
#y = x



# (1)


# (2) Normalise results with respect to USA levels (5th element of first row).

tfp_norm = np.divide(tfpmatrix[1:],np.float32(tfpmatrix[5]))
gdp_norm = np.divide(gdpmatrix[1:],np.float32(gdpmatrix[5]))
E_norm = np.divide(Ematrix[1:],np.float32(Ematrix[5]))

# Graph (1): 'Model TFP and entry costs'


#plt.suptitle('Model TFP and entry costs')
#plt.ylim(np.min(tfp_norm)*0.9,np.max(tfp_norm)*1.1)
#plt.xlim(0,np.max(y[0])*1.1)
#plt.xlabel('Entry costs')
#plt.ylabel('TFP (normalised)')
#plt.legend()
#plt.scatter(y[0],tfp_norm, c='g',marker='.')
#plt.show()

# Graph (2): 'Model GNIpc vs. Real GNIpc'


#plt.suptitle('Predicted and real income differences')
#plt.ylabel('Model GNIpc (normalised)')
#plt.xlabel('Observed GNIpc (normalised)')
## Plot per groups:
#plt.scatter(GNIpc,gdp_norm, c='b',marker='.')
##plt.scatter(GNIpc,gdp_norm, c='b',marker='.')
##plt.scatter(GNIpc,gdp_norm, c='b',marker='.')
##plt.scatter(GNIpc,gdp_norm, c='b',marker='.')
#plt.plot(x,y,'-r') # Don't forget the 45 degree line!!!
#plt.xlim(0,np.max(GNIpc)*1.05)
#plt.ylim(0.9*(np.min(gdp_norm)),np.max(gdp_norm)*1.05)
#plt.legend(loc='4',fancybox=True,shadow=True)
#plt.show()

























