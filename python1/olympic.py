import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

import pandas as pd
athlete = pd.read_csv("C:/Users/Spm/Downloads/olympicd/athlete_events.csv")
region =  pd.read_csv("C:/Users/Spm/Downloads/olympicd/noc_regions.csv")
print(athlete)
print(region)
athleteregion = athlete.merge(region,how='left', on = 'NOC')
print(athleteregion)
athleteregions = athleteregion.rename(columns={'region': 'Region', 'notes': 'Notes'})
print(athleteregions)
print(athleteregions.info())
print(athleteregions.describe())
nan_values = athleteregions.isna() # if any value is null
nan_columns = nan_values.any() # for if one column is null
print(nan_columns)
print(athleteregions.isnull().sum())
print(athleteregions.query('Team =="India"'))# show data for team india
print(athleteregions.query('Team =="Japan"'))
top_10_country = athleteregions.Team.value_counts().sort_values(ascending=False).head(10)# top country in olympics
print(top_10_country) 

# plot for the top 10 country
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming top_10_country is a pandas Series or DataFrame with the data to be plotted

# Overall Participation by Country
plt.figure(figsize=(10, 4))
plt.title('Overall Participation by Country')
sns.barplot(x=top_10_country.index, y=top_10_country, palette='Set2')
plt.show()

# Age distribution of the participants
plt.figure(figsize=(10, 4))
plt.title("Age distribution of the athletes")
plt.xlabel('Age')
plt.ylabel('Number of participants')
plt.hist(athleteregions.Age, bins=np.arange(10, 80, 2), color='red', edgecolor='black')
plt.show()

#winter sports
winter_sports = athleteregions[athleteregions.Season == 'Winter'].Sport.unique()
print(winter_sports)

summer_sports = athleteregions[athleteregions.Season == 'Summer'].Sport.unique()
print(summer_sports)

#Male and Female participants
gender_count = athleteregions.Sex.value_counts()
print(gender_count)

# pie chart for male and female athletes
plt.figure(figsize=(10,6))
plt.title('Gender Distrubution')
plt.pie(gender_count,labels=gender_count.index,autopct='%1.1f%%',startangle=150)
plt.show()
#Total medals
medal_count = athleteregions.Medal.value_counts()
print(medal_count)

# Total numbers of female in each olympic
female_count = athleteregions[(athleteregions.Sex == 'F') & (athleteregions.Season =='Summer')][['Sex','Year']]
female_count = female_count.groupby('Year').count().reset_index()                                                                                               
print(female_count)
women_count = athleteregions[(athleteregions.Sex == 'F')&(athleteregions.Season == 'Summer')]


# Women participation
sns.set(style="dark")
plt.figure(figsize=(15,8))
sns.countplot(x='Year',data=women_count,palette="Spectral")
plt.title('Women participates')
plt.show()

#Gold medal athlete
gold_medal = athleteregions[(athleteregions.Medal == 'Gold')]
print(gold_medal)
medal_gold = gold_medal['ID'][gold_medal['Age']>60].count()
print(medal_gold)
sports_event = gold_medal['Sport'][gold_medal['Age']>60]
print(sports_event)
# Meadal above 60
plt.figure(figsize=(8,4))
plt.title("Gold medal above 60")
sns.countplot(sports_event)
plt.tight_layout()
plt.show()
# Gold medal for each country

gold_medals = athleteregions[athleteregions['Medal'] == 'Gold']

# Counting gold medals for each country
gold_medal_counts = gold_medals['Region'].value_counts().head(10)

# Displaying the result
print(gold_medal_counts)

# Plotting the top 10 countries with the highest count of gold medals
plt.figure(figsize=(10, 6))
sns.barplot(x=gold_medal_counts.index, y=gold_medal_counts, palette='Set2')
plt.title('Top 10 Countries with the Highest Count of Gold Medals')
plt.xlabel('Country')
plt.ylabel('Count of Gold Medals')
plt.show()

# Latest olympic medal

latest_oly = athleteregions.Year.max()
team = athleteregions[(athleteregions.Year == latest_oly) & (athleteregions.Medal == 'Gold')].Team
teams = team.value_counts().head(10)                     
print(teams) 

# Plotting the top 10 countries with the highest count of gold medals in 2016
plt.figure(figsize=(10, 6))
sns.barplot(x=teams.index, y=teams, palette='Set2')
plt.title('Top 10 Countries with the Highest Count of Gold Medals in 2016')
plt.xlabel('Country')
plt.ylabel('Count of Gold Medals')
plt.show()









