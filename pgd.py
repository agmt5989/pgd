# -*- coding: utf-8 -*-
"""
Spyder Editor

This is Mike Ajala.
"""
#import sys
#print(sys.executable)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import stats
import csv
import pylab as pl
import matplotlib.mlab as mlab
import pandas as pd
import seaborn as sns

with open('/home/cslab/data_new.csv', 'r') as f:
    reader = csv.reader(f)
    mylist = list(reader)
    
np_list = np.array(mylist)
pgd = np.column_stack(np_list)

# Individual Features in the Dataset
id = np.array(list(map(int, pgd[0]))) # Unique Numeric Identifiers 
test_name = pgd[1] # Code name of test participants
num_subj = pgd[2] # Number of subjects {4}
attempt = np.array(list(map(int, pgd[3]))) # Number of attempts {40}
score = score = np.array(list(map(int, pgd[4]))) # Score {40}
name = pgd[5] # Real names
sex = pgd[6] # Sexes
state = pgd[7] # States of Origin
age = np.array(list(map(int, pgd[8]))) # Ages
phone = pgd[9] # Phone Numbers
email = pgd[10] # Email addresses
kin_phone = pgd[11]
kin_email = pgd[12]
kin_add = pgd[13]
subj1 = pgd[14] # UTME English
subj1_score = np.array(list(map(int, pgd[15]))) # Score {100}
subj2 = pgd[16] # UTME Second Subject
subj2_score = np.array(list(map(int, pgd[17]))) # Score {100}
subj3 = pgd[18]
subj3_score = np.array(list(map(int, pgd[19]))) # Score {100}
subj4 = pgd[20]
subj4_score = np.array(list(map(int, pgd[21]))) # Score {100}
jamb_total = np.array(list(map(int, pgd[22]))) # Total UTME Score {400}
faculty = pgd[23] # Faculty {8 Types}
course = pgd[24] # Course of Study {51 Types}
lga = pgd[25] # Local Government Area {205 Types}

## PostUTME Score Plots
# Score Array
print (score)

# Line Plot
plt.plot(id, score)
plt.title('Candidates\' individual scores')
plt.xlabel('Candidates')
plt.ylabel('Scores')
plt.show()

# Scatter Plot
plt.scatter(id, score)
plt.title('Candidates\' individual scores')
plt.xlabel('Candidates')
plt.ylabel('Scores')
plt.show()

# Dynamic bins for histogram with discrete values
d = np.diff(np.unique(score)).min()
score_lower = score.min() - float(d) / 2
score_upper = score.max() + float(d) / 2
bins = np.arange(score_lower, score_upper + d, d)

# Yellow-centered histogram
N, ibins, patches = plt.hist(score, bins=bins, edgecolor='white', linewidth=0.85)
fracs = N / N.max()
norm = colors.Normalize(fracs.min(), fracs.max())
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

# Colored Histogram
fig, pg = plt.subplots()
N, bins, patches = pg.hist(score, bins=bins, edgecolor='white', linewidth=1)
palette = pl.get_cmap('jet', len(patches))
for i in range(len(patches)):
    patches[i].set_facecolor(palette(i))

plt.show()

## UTME Score Plots
# Score Array
print (jamb_total)

# Line Plot
plt.plot(id, jamb_total)
plt.title('Candidates\' individual scores')
plt.xlabel('Candidates')
plt.ylabel('Scores')
plt.show()

# Scatter Plot
plt.scatter(id, jamb_total)
plt.title('Candidates\' individual scores')
plt.xlabel('Candidates')
plt.ylabel('Scores')
plt.show()

# Dynamic bins for histogram with discrete values
d = np.diff(np.unique(jamb_total)).min()
score_lower = jamb_total.min() - float(d) / 2
score_upper = jamb_total.max() + float(d) / 2
bins = np.arange(score_lower, score_upper + d, d)
#bins /= 2

# Yellow-centered histogram
N, ibins, patches = plt.hist(jamb_total, bins=bins, edgecolor='white', linewidth=0.85)
fracs = N / N.max()
norm = colors.Normalize(fracs.min(), fracs.max())
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

# Colored Histogram
fig, pg = plt.subplots()
N, bins, patches = pg.hist(jamb_total, bins=bins, edgecolor='white', linewidth=1)
palette = pl.get_cmap('jet', len(patches))
for i in range(len(patches)):
    patches[i].set_facecolor(palette(i))

plt.show()


## Age Distribution

# Age Bins
age_d = np.diff(np.unique(age)).min()
age_lower = age.min() - float(age_d) / 2
age_upper = age.max() + float(age_d) / 2
age_bin = np.arange(age_lower, age_upper + age_d, age_d)

# Colored Histogram
fig, age_dist = plt.subplots()
N, bins, patches = age_dist.hist(age, bins=age_bin, edgecolor='white', linewidth=1)
palette = pl.get_cmap('jet', len(patches))
for i in range(len(patches)):
    patches[i].set_facecolor(palette(i))




## Preliminary data insights
# Dataset Size
print(id.shape)
print(len(id))

# PostUTME Scores
min_score = np.min(score)
max_score = np.max(score)
mean_score = np.mean(score)
median_score = np.median(score)
modal_score = int(stats.mode(score)[0])
std_score = np.std(score)
# Correlation Coefficient between UTME and PostUTME
correlation = np.corrcoef(score, jamb_total)
# Number of passes
postutme_pass = score[score >= 20]
postutme_fail = score[score < 20]

# UTME Scores
min_utme = np.min(jamb_total)
max_utme = np.max(jamb_total)
mean_utme = np.mean(jamb_total)
median_utme = np.median(jamb_total)
modal_utme = int(stats.mode(jamb_total)[0])
std_utme = np.std(jamb_total)

# Ages
min_age = np.min(age)
max_age = np.max(age)
mean_age = np.mean(age)
median_age = np.median(age)
modal_age = int(stats.mode(age)[0])
std_age = np.std(age)


unique_age = np.unique(age, return_counts=True)
unique_age_freq = unique_age[1]
unique_age = unique_age[0]

funny_age = np.unique(score[age==14], return_counts=False)
print(funny_age)


# Score of people who scored 14
score_14 = score[age==14]
print(score_14)
plt.hist(score_14, 27)
plt.show()

score_14_mean = np.mean(score_14)
print(score_14_mean)


plt.scatter(id[age==16], score[age==16], alpha=0.9, color='red')
plt.show()


# Analysis based on age groups
cls = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'firebrick', 'darkslateblue', 'gold', 'deepskyblue', 'yellow', 'cornflowerblue', 'darkseagreen', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'firebrick', 'thistle', 'gold', 'deepskyblue', 'yellow', 'cornflowerblue', 'darkseagreen', 'aquamarine']

indi = 0
age_group = []
ave_ages = []
for bracket in np.unique(age):
    age_group.append(score[age == bracket])
    ave_ages.append(np.mean(score[age == bracket])) 

age_group = np.array(age_group)
# print([unique_age, age_group])
plt.figure(figsize=(30,4))
plt.xticks(unique_age)
plt.scatter(unique_age, ave_ages, c=cls, s=unique_age_freq*3.5, alpha=0.75, marker='.', linewidths=0)
plt.show()

# Line Graph for unique ages
plt.figure(figsize=(30,4))
plt.xticks(unique_age)
plt.plot(unique_age, ave_ages)
plt.show()

# Analysis based on States
unique_state = np.unique(state, return_counts=True)
state_freq = unique_state[1]
unique_state = unique_state[0]
state_score = []
ave_state = []


for this_state in unique_state:
    state_score.append(score[state == this_state])
    ave_state.append(np.mean(score[state == this_state]))

# Scatterplot
plt.figure(figsize=(30,4))
plt.xticks(rotation='vertical')
plt.yticks(np.rint(ave_state))
plt.scatter(unique_state, ave_state, c=cls, s=state_freq*2.5, alpha=0.68, marker='o')
plt.show()

# Line Graph
plt.figure(figsize=(30,4))
plt.xticks(rotation='vertical')
plt.yticks(np.rint(ave_state))
plt.plot(unique_state, ave_state)
plt.show()


# Correlation between PUTME Scores and UTME
unique_putme = np.unique(score)
utme_avgs = []

for single_score in unique_putme:
    utme_avgs.append(np.mean(jamb_total[score == single_score]))
    
plt.plot(unique_putme, utme_avgs)
plt.show()
    
plt.scatter(unique_putme, utme_avgs)
plt.show()



##############################

n_groups = 4
means_frank = (90, 55, 40, 65)
means_guido = (85, 62, 54, 20)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, means_frank, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Frank')
 
rects2 = plt.bar(index + bar_width, means_guido, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Guido')
 
plt.xlabel('Person')
plt.ylabel('Scores')
plt.title('Scores by person')
plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
plt.legend()
 
plt.tight_layout()
plt.show()

##############################

##############################



import plotly.plotly as py
import plotly.tools as tls
tls.set_credentials_file(username='imyke', api_key='kf4pLdaRwl0W9ogABSjp')


import matplotlib.pyplot as plt
import numpy as np

mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)

N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = ax.bar(ind, menMeans, width, color=(0.2588,0.4433,1.0))
p2 = ax.bar(ind, womenMeans, width, color=(1.0,0.5,0.62),
             bottom=menMeans)
ax.set_ylabel('Scores')
ax.set_xlabel('Groups')
ax.set_title('Scores by group and gender')

ax.set_xticks(ind + width/2.)
ax.set_yticks(np.arange(0, 81, 10))
ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))

plotly_fig = tls.mpl_to_plotly( mpl_fig )

# For Legend
plotly_fig["layout"]["showlegend"] = True
plotly_fig["data"][0]["name"] = "Men"
plotly_fig["data"][1]["name"] = "Women"
py.iplot(plotly_fig, filename='stacked-bar-chart')

##############################

##############################


import numpy as np
import matplotlib.pyplot as plt


N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans, width, yerr=menStd)
p2 = plt.bar(ind, womenMeans, width,
             bottom=menMeans, yerr=womenStd)

plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()


# Analysis based on Faculties
N = 9
unique_fac = np.unique(faculty, return_counts=True)
fac_freq = unique_fac[1]
unique_fac = unique_fac[0]
ind = np.arange(N)
width = 0.35  

putme_pass = score >= 20

fac_pass = []

for single_fac in unique_fac:
    fac_pass.append(len(np.select([putme_pass, faculty == single_fac], score)))


p1 = plt.bar(ind, menMeans, width, yerr=menStd)
p2 = plt.bar(ind, womenMeans, width,
             bottom=menMeans, yerr=womenStd)


##############################

# LGAs
unique_lgas = np.unique(lga, return_counts=True)
lga_freq = unique_lgas[1]
unique_lgas = unique_lgas[0]
lga_score = []















# Garbage
plt.hist(faculty, 8)
plt.show()


print(median_score, min_score, max_score, mean_score)
print(max_age)



import mysql.connector    
cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='pgd')

try:
   cursor = cnx.cursor()
   cursor.execute("""
      select name, score from data_summary limit 10
   """)
   result = cursor.fetchall()
   print (result)
finally:
    cnx.close()