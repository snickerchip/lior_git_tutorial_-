#!/usr/bin/env python
# coding: utf-8

# # Checkpoint 0 

# These exercises are a mix of Python and Pandas practice. Most should be no more than a few lines of code! 

# In[ ]:


# here is a Python list:

a = [1, 2, 3, 4, 5, 6]


# In[ ]:


# get a list containing the last 3 elements of a
# Yes, you can just type out [4, 5, 6] but we really want to see you demonstrate you know how to do that in Python
a[3:6]


# In[ ]:


# create a list of numbers from 1 to 100
list(range(1,101))


# In[ ]:


# now get a list with only the even numbers between 1 and 100
# you may or may not make use of the list you made in the last cell
list(range(0,102,2))


# In[11]:


# write a function that takes two numbers as arguments
# and returns the first number divided by the second
def divide(num1, num2):
    print("Number 1:", num1)
    print("Number 2:", num2)
    division = num1/num2
    
    return division


res = divide(4, 2)
print(res)


# In[26]:


# write a function that takes a string as input
# and return that string in all caps
def words(oi_im_butcher):
    print(oi_im_butcher.upper())
    
    return words

res = words("oi_im_butcher")
print(res)


# In[27]:


# optional challenge - fizzbuzz
# you will need to use both iteration and control flow 
# go through all numbers from 1 to 100 in order
# if the number is a multiple of 3, print fizz
# if the number is a multiple of 5, print buzz
# if the number is a multiple of 3 and 5, print fizzbuzz and NOTHING ELSE
# if the number is neither a multiple of 3 nor a multiple of 5, print the number
print("naur")


# In[33]:


# create a dictionary that reflects the following menu pricing (taken from Ahmo's)
# Gyro: $9 
# Burger: $9
# Greek Salad: $8
# Philly Steak: $10

Ahmos_pricing = {"Gyro":9.00, "Burger":9.00, "Greek Salad":8.00, "Philly Steak":10.00}
print(Ahmos_pricing.items())


# In[36]:


# load in the "starbucks.csv" dataset
# refer to how we read the cereal.csv dataset in the tutorial
import pandas as pd
df = pd.read_csv("../data/starbucks.csv")
type(df)

df


# In[110]:


# output the calories, sugars, and protein columns only of every 40th row.
df.iloc[::40]

df_ = df.set_index('beverage_category')
df_.head()

df.iloc[::40, [3, 11, 12]]


# In[112]:


# select all rows with more than and including 400 calories
df[df["calories"] >= 400]


# In[113]:


# select all rows whose vitamin c content is higher than the iron content

df[df["vitamin c"] > df["iron"]]


# In[119]:


# create a new column containing the caffeine per calories of each drink

df["caffeine_per_calories"] = df["caffeine"] / df["calories"]
df.head()


# In[130]:


# what is the average calorie across all items?
df["calories"].mean()


# In[133]:


# how many different categories of beverages are there?
print(df["beverage_category"].unique())
print(df["beverage_category"].nunique())


# In[134]:


# what is the average # calories for each beverage category?
beverage_category = df.groupby("beverage_category")
beverage_category["calories"].mean()


# In[136]:


# plot the distribution of the number of calories in drinks with a histogram

df["calories"].plot.hist(edgecolor='black', alpha=0.8, title="Calories In The Drinks")


# In[138]:


# plot calories against total fat with a scatterplot

df.plot.scatter(x="calories", y="total fat", title="calories vs total fat content")


# In[140]:


print("i feel powerful because i can code!!!!!!")


# In[ ]:




