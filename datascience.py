import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly.express as px

df = pd.read_csv("C:\\Users\\Spm\\Downloads\\jobs_in_data.csv")
print(df)
print(df.columns)

labels = df["experience_level"].value_counts().index
sizes = df["experience_level"].value_counts()
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99',"pink","yellow"]
plt.figure(figsize = (10,16))
plt.pie(sizes, labels=labels, rotatelabels=False, autopct='%1.1f%%',colors=colors,shadow=True, startangle=45)
plt.title('States',color = 'red',fontsize = 15)
plt.show()

plt.figure(figsize=(16,4))
sns.countplot(data=df,x="work_year",palette="icefire")
plt.xticks(fontsize=10,rotation=50)
plt.xlabel("JOB Year",fontsize=12,color="RED")
plt.ylabel("COUNT",fontsize=10,color="RED")
plt.title("Working Year",fontsize=10,color="RED")
plt.show()

plt.figure(figsize=(16,6))
sns.countplot(data=df,x="job_category",palette="icefire")
plt.xticks(fontsize=10,rotation=50)
plt.xlabel("JOB CATEGORIES",fontsize=12,color="RED")
plt.ylabel("COUNT",fontsize=10,color="RED")
plt.title("CATEGORIES BY JOB COUNT",fontsize=10,color="RED")
plt.show()



fig=px.bar(df.groupby('job_title',as_index=False)['salary_in_usd'].max().sort_values(by='salary_in_usd',ascending=False).head(10),x='job_title',y='salary_in_usd',color='job_title',labels={'job_title':'job title','salary_in_usd':'salary in usd'},template='ggplot2',text='salary_in_usd',title='<b> Top 10 Highest Paid Roles in Data Science')
fig.show()

z=df.groupby('job_title',as_index=False)['salary_in_usd'].mean().sort_values(by='salary_in_usd',ascending=False)
z['salary_in_usd']=round(z['salary_in_usd'],2)
fig=px.bar(z.head(10),x='job_title',y='salary_in_usd',color='job_title',labels={'job_title':'job title','salary_in_usd':'avg salary in usd'},text='salary_in_usd',template='seaborn',title='<b> Top 10 Roles in Data Science based on Average Pay')
fig.update_traces(textfont_size=8)
fig.show()
fig = px.funnel(
    df.groupby('company_location', as_index=False)['experience_level'].count().sort_values(by='experience_level', ascending=False).head(15),
    y='company_location',
    x='experience_level',
    color_discrete_sequence=['yellow'],
    labels={'experience_level': 'count'},
    template='seaborn',
    title='<b>Top 15 Countries having maximum Data Science Jobs')

fig.show()

fig=px.pie(df.groupby('experience_level',as_index=False)['salary_in_usd'].count().sort_values(by='salary_in_usd',ascending=False).head(10),names='experience_level',values='salary_in_usd',color='experience_level',hole=0.7,labels={'experience_level':'Experience level ','salary_in_usd':'count'},template='ggplot2',title='<b>Total Jobs Based on Experience Level')
fig.update_layout(title_x=0.5,legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1))
fig.show()

fig=px.pie(df.groupby('work_setting',as_index=False)['salary_in_usd'].count().sort_values(by='salary_in_usd',ascending=False).head(10),names='work_setting',values='salary_in_usd',color='work_setting',hole=0.7,labels={'remote_ratio':'work_setting','salary_in_usd':'count'},template='plotly',title='<b> Remote Ratio')
fig.update_layout(title_x=0.5)
fig.show()

