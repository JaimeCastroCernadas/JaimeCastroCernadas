#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pyspark


# In[1]:


get_ipython().system('pip install -q findspark')


# # 1. START THE SPARK SESSION

# In[2]:


from pyspark.sql import SparkSession
name="Crew member regression"


spark = SparkSession.builder.appName(name).master("local[*]").getOrCreate()


# In[4]:


spark


# # 2. LOAD DATA

# In[3]:


cruise= "/Users/jaimecastro/Downloads/cruise_ship_info.csv"
cruise_df=spark.read.format("csv")\
                    .options(header=True, inferschema=True,delimiter=",")\
                    .load(cruise)


# # 3.BASIC DATA ANALYSIS

# In[17]:


cruise_df.printSchema()


# In[31]:


cruise_df.toPandas().describe()


# In[18]:


cruise_df.toPandas().head()


# In[25]:


cruise_df.toPandas().tail()


# Null values

# In[8]:


from pyspark.sql.functions import when, count, col

cruise_df.select([count(when(col(c).isNull(), c)).alias(c) for c in cruise_df.columns]).show() 



# In[59]:


cruise_df.select("Cruise_line").distinct().collect()


# # EDA

# In[71]:


cruise_df.createOrReplaceTempView('cruise_df')


# Target variable: Crew

# In[63]:


import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

cruise_df_pandas= cruise_df.toPandas()

plt.figure(figsize=(10, 6))
sns.histplot(cruise_df_pandas["crew"], kde=True, bins=30)
plt.title("Distribution of crew Sizes")
plt.xlabel("Crew Size")
plt.ylabel("Frequency")
plt.show()

cruise_df_pandas["crew"].describe()


# In[72]:


spark.sql("""
    SELECT *
    FROM cruise_df
    WHERE crew > 17.5
"""
).toPandas()


# In[36]:


spark.sql("""
    SELECT *
    FROM cruise_df
    WHERE Cruise_line = 'Carnival' OR Cruise_line = 'Royal_Caribbean'
    ORDER BY crew DESC
"""
).toPandas()


# The Royal_Caribbean outlier seems to be a normal value, since it´s Tonnage, Passengers, andcabins a

# Check correlations between numeric variables and the crew

# In[67]:


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# Select only the numeric columns.
numeric_columns = ["crew", "Tonnage", "Age", "passengers","cabins","passenger_density","length"]
#Assemble the vector in order to use the MLIB library.
#I assemble the vector: The columns i want to include and how will this columns be named in the vector
assembler_numeric = VectorAssembler(inputCols=numeric_columns, outputCol="features")
#the transform method is combining the inputs from the assembler columns and creating a single vector
vector_cruise_data_numeric = assembler_numeric.transform(cruise_df)

#Stores the correlation matrix of the features using the correlation.corr method
correlation = Correlation.corr(vector_cruise_data_numeric, "features")
#Stores the actual value of correlation matrix into an array
correlation_matrix = correlation.head()[0].toArray()

#Converts those values to a pandas DataFrame for plotting.
correlation_df=pd.DataFrame(correlation_matrix, columns=numeric_columns,index=numeric_columns)

#Creating the heatmap to represent the correlation data.
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_df, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})

# Title and labels
plt.title("Correlation Matrix Heatmap")
plt.show()

#Explanation of the correlation results.
#Iterate over each colum name
for column in numeric_columns:
    if column != "crew":  # Skip the 'crew' column itself
#         corr_value = correlation_df.loc[column, "crew"]  #Extract the correlation value
        if (0.4 <= corr_value < 0.7) or (-0.7 < corr_value <= -0.4):
            print(f"{column} is mildly correlated with crew (Correlation: {corr_value:.2f})")
        elif (corr_value >= 0.7) or (corr_value <= -0.7):
            print(f"{column} is highly correlated with crew (Correlation: {corr_value:.2f})")
        elif (0 < corr_value < 0.4) or (-0.4 < corr_value < 0):
            print(f"{column} is poorly correlated with crew (Correlation: {corr_value:.2f})")


# Since passenger density is lowly correlated with crew, I won´t use that feature in my prediction.
# 
# For the numeric values, I will use the Tonnage, Age, passengers, cabins and length of the ship, since they are correlated with the number of crew need for the ship

# Relation of crew with the cruise_line

# In[43]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Cruise_line', y='crew', data=cruise_df_pandas)
plt.title("Crew Size Distribution by Cruise Line")
plt.xlabel("Cruise Line")
plt.ylabel("Crew Size")
plt.xticks(rotation=45)
plt.show()


# There seems to be some outliers in most of the cruise lines. 

# In[84]:


#Calculate mean crew size grouped by cruise_line
cruise_summary = cruise_df_pandas.groupby('Cruise_line')['crew'].agg(
    mean_crew='mean',
    q75_crew=lambda x: x.quantile(0.75)
).reset_index()

print(cruise_summary)


# Use string indexer to convert cruise_line to a numerical value

# In[5]:


from pyspark.ml.feature import StringIndexer

indexer= StringIndexer(inputCol="Cruise_line",outputCol="Indexed_line")
IndexerModel= indexer.fit(cruise_df)

indexed_cruise_df= IndexerModel.transform(cruise_df)
print(indexed_cruise_df.head())


# # Let´s begin our Regression model

# # 1. Assemble the features

# In[41]:


from pyspark.ml.feature import VectorAssembler

data_assembler= VectorAssembler(inputCols=["Tonnage", "Age", "passengers","cabins","length","Indexed_line"],outputCol="features")
transformed_cruise_data=data_assembler.transform(indexed_cruise_df)

transformed_cruise_data.show()


# In[42]:


transformed_cruise_data.select("features").head(5)


# # SELECT AND SPLIT OUR DATA TO CREATE THE MODEL

# In[43]:


final_data= transformed_cruise_data.select(["features","crew"])

train_data, test_data= final_data.randomSplit([0.8,0.2])


# In[44]:


print("Training data distribution:")
train_data.show()

print("Test data distribution:")
test_data.show()


# # CREATE THE ESTIMATOR

# In[45]:


from pyspark.ml.regression import LinearRegression
#this is the algorithm as it is.
linear_reg_est=LinearRegression(featuresCol="features",labelCol="crew")

#Because I didn´t call my target variable "label", which is the default value, I have to/
#specify it´s name in labelCol. The rest of the parameters will use their default values, includin featuresCol.


# # CREATE THE MODEL

# Training the model

# In[46]:


#Fit the model (the estimator) with the training data.
linear_reg_model=linear_reg_est.fit(train_data)


# In[47]:


print("Coefficients: %s" % str(linear_reg_model.coefficients), 
      "\nIntercept: %s" % str(linear_reg_model.intercept), 
      "\nMean Squared Error: %s" % str(linear_reg_model.summary.meanSquaredError))


# Doing the predictions

# In[48]:


linear_reg_prediction=linear_reg_model.transform(test_data)


# # EVALUATION OF THE MODEL

# In[49]:


from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="crew")
print(evaluator.explainParams())


# In[52]:


r2_linear_reg=evaluator.evaluate(linear_reg_prediction, { evaluator.metricName: "r2" })
print("R^2 metric for the model:", r2_linear_reg)


# In[ ]:




