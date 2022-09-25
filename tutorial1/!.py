#!/usr/bin/env python
# coding: utf-8

# # Checkpoint 1

# Reminder: 
# 
# - You are being evaluated for compeletion and effort in this checkpoint. 
# - Avoid manual labor / hard coding as much as possible, everything we've taught you so far are meant to simplify and automate your process.

# We will be working with the same `states_edu.csv` that you should already be familiar with from the tutorial.
# 
# We investigated Grade 8 reading score in the tutorial. For this checkpoint, you are asked to investigate another test. Here's an overview:
# 
# * Choose a specific response variable to focus on
# >Grade 4 Math, Grade 4 Reading, Grade 8 Math
# * Pick or create features to use
# >Will all the features be useful in predicting test score? Are some more important than others? Should you standardize, bin, or scale the data?
# * Explore the data as it relates to that test
# >Create at least 2 visualizations (graphs), each with a caption describing the graph and what it tells us about the data
# * Create training and testing data
# >Do you want to train on all the data? Only data from the last 10 years? Only Michigan data?
# * Train a ML model to predict outcome 
# >Define what you want to predict, and pick a model in sklearn to use (see sklearn <a href="https://scikit-learn.org/stable/modules/linear_model.html">regressors</a>.
# * Summarize your findings
# >Write a 1 paragraph summary of what you did and make a recommendation about if and how student performance can be predicted
# 
# Include comments throughout your code! Every cleanup and preprocessing task should be documented.
# 
# Of course, if you're finding this assignment interesting (and we really hope you do!), you are welcome to do more than the requirements! For example, you may want to see if expenditure affects 4th graders more than 8th graders. Maybe you want to look into the extended version of this dataset and see how factors like sex and race are involved. You can include all your work in this notebook when you turn it in -- just always make sure you explain what you did and interpret your results. Good luck!

# <h2> Data Cleanup </h2>
# 
# Import `numpy`, `pandas`, and `matplotlib`.
# 
# (Feel free to import other libraries!)

# In[36]:


import numpy as np
import pandas as pd
import matplotlib as plte
from matplotlib import pyplot as plt


# Load in the "states_edu.csv" dataset and take a look at the head of the data

# In[6]:


df = pd.read_csv('../data/states_edu.csv')
df.head()


# You should always familiarize yourself with what each column in the dataframe represents. Read about the states_edu dataset here: https://www.kaggle.com/noriuk/us-education-datasets-unification-project

# Use this space to rename columns, deal with missing data, etc. _(optional)_

# In[7]:


print('nah')


# <h2>Exploratory Data Analysis (EDA) </h2>

# Chosen Outcome Variable for Test: *ENTER YOUR CHOICE HERE*

# How many years of data are logged in our dataset? 

# In[8]:


df["YEAR"].nunique()


# Let's compare Michigan to Ohio. Which state has the higher average outcome score across all years?

# In[10]:


states = df.groupby("STATE")
avg = states["AVG_READING_8_SCORE"].mean()
if (avg.loc["MICHIGAN"] > avg.loc["OHIO"]):
    print("michigan has higher score")
else:
    print("ohio has higher score")


# Find the average for your outcome score across all states in 2019

# In[18]:


years = df.groupby("YEAR")
avg = years["AVG_READING_8_SCORE"].mean()
avg.loc[2019]


# Find the maximum outcome score for every state. 
# 
# Refer to the `Grouping and Aggregating` section in Tutorial 0 if you are stuck.

# In[19]:


states["AVG_READING_8_SCORE"].max()


# <h2> Feature Engineering </h2>
# 
# After exploring the data, you can choose to modify features that you would use to predict the performance of the students on your chosen response variable. 
# 
# You can also create your own features. For example, perhaps you figured that maybe a state's expenditure per student may affect their overall academic performance so you create a expenditure_per_student feature.
# 
# Use this space to modify or create features.

# In[28]:


df["AVERAGE_MATH_SCORES"] = (df["AVG_MATH_8_SCORE"] + df["AVG_MATH_4_SCORE"])/2
df["AVERAGE_MATH_SCORES"]


# Feature engineering justification: **I combined math scores of grades 4 and 8**

# <h2>Visualization</h2>
# 
# Investigate the relationship between your chosen response variable and at least two predictors using visualizations. Write down your observations.
# 
# **Visualization 1 aka make a graph**

# In[167]:


df.AVERAGE_MATH_SCORES.plot.hist(title="Distribution of Math Scores", edgecolor="black")


# **a lot of concentration around 250-265 range**

# **Visualization 2**

# In[187]:


df.groupby('STATE')["AVERAGE_MATH_SCORES"].mean().plot()
plt.ylabel('SCORE')
plt.title('Math Scores By State')
# this is kinda messed up but oh well


# **hey i didn't get an error message did i**

# <h2> Data Creation </h2>
# 
# _Use this space to create train/test data_

# In[169]:


from sklearn.model_selection import train_test_split


# In[170]:


# X =
# y = 
X = df[['YEAR']].dropna()
y = df.loc[X.index]["AVERAGE_MATH_SCORES"]
y.fillna(y.mean(), inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0) 


# In[171]:


# X_train, X_test, y_train, y_test = train_test_split(
#      X, y, test_size=, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0) 


# <h2> Prediction </h2>

# ML Models [Resource](https://medium.com/@vijaya.beeravalli/comparison-of-machine-learning-classification-models-for-credit-card-default-data-c3cf805c9a5a)

# In[172]:


# create your model here
# model = 
model = LinearRegression()


# In[173]:


# import your sklearn class here
from sklearn.linear_model import LinearRegression


# In[174]:


model.fit(X_train, y_train)


# In[175]:


print(model.intercept_)
print(model.coef_)


# ## Evaluation

# Choose some metrics to evaluate the performance of your model, some of them are mentioned in the tutorial.

# In[176]:


model.score(X_test, y_test)


# We have copied over the graphs that visualize the model's performance on the training and testing set. 
# 
# Change `col_name` and modify the call to `plt.ylabel()` to isolate how a single predictor affects the model.

# In[183]:


# col_name = 'COLUMN NAME OF ONE PREDICTOR'

# f = plt.figure(figsize=(12,6))
# plt.scatter(X_train[col_name], y_train, color = "red")
# plt.scatter(X_train[col_name], model.predict(X_train), color = "green")

# plt.legend(['True Training','Predicted Training'])
# plt.xlabel(col_name)
# plt.ylabel('NAME OF THE PREDICTOR')
# plt.title("Model Behavior On Training Set")
col_name = 'AVERAGE_MATH_SCORE'

f = plt.figure(figsize=(12,6))
plt.scatter(X_train[plt.ylabel('YEAR')], y_train, color = "red")
plt.scatter(X_train[col_name], model.predict(X_train), color = "green")

plt.legend(['True Training','Predicted Training'])
plt.xlabel(col_name)
plt.ylabel('YEAR')
plt.title("Model Behavior On Training Set")

print("I have been trying many different combinations here are the fruits of my labor")


# In[125]:


# col_name = 'COLUMN NAME OF ONE PREDICTOR"

# f = plt.figure(figsize=(12,6))
# plt.scatter(X_test[col_name], y_test, color = "blue")
# plt.scatter(X_test[col_name], model.predict(X_test), color = "black")

# plt.legend(['True testing','Predicted testing'])
# plt.xlabel(col_name)
# plt.ylabel('NAME OF THE PREDICTOR')
# plt.title("Model Behavior on Testing Set")

col_name = 'AVG_MATH_4_SCORE'

f = plt.figure(figsize=(12,6))
plt.scatter(X_train[col_name], y_train, color = "red")
plt.scatter(X_train[col_name], model.predict(X_train), color = "green")

plt.legend(['True Training','Predicted Training'])
plt.xlabel(col_name)
plt.ylabel('AVERAGE_MATH_SCORES')
plt.title("Model Behavior On Training Set")


# <h2> Summary </h2>

# **I attemped something here, very much attemped. I'm not entirely sure what's happening, but i'm sure i'll figure it out. I found nothing out from my data predictions or creations, but from my visualizations I found out two things. 1, the average math score across all ages is mostly concentrarted in the 250-265 range; and 2, not all states have equal math tests scores. My models, I would say, did not perform well in the slightest, at least the one i tried to create on my own. The one I copied over works, but only because I didn't attempt to create it myself.**

# In[ ]:




