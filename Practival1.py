import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import load_dataset
#titanic dataset
data = pd.read_csv("train.csv")
#tips dataset
tips = load_dataset("tips")

data['Sex'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()

plt.hist(data['Age'], bins=5)
plt.show()

sns.distplot(data['Age']) 
plt.show()

sns.scatterplot(data=tips, x="total_bill", y="tip")
# Show the plot
plt.show()

sns.scatterplot(data=tips, x="total_bill", y="tip", hue=tips["sex"])
plt.show()

sns.scatterplot(data=tips, x="total_bill", y="tip", hue=tips["sex"], style=tips['smoker'])
plt.show()

sns.barplot(data=data, x="Pclass", y="Age")
plt.show()

sns.barplot(data=data, x="Pclass", y="Age", hue = data["Sex"])
plt.show()

sns.boxplot(data=data, x="Sex", y="Age")
plt.show()

sns.boxplot(data=data, x="Sex", y="Age", hue = data["Survived"])
plt.show()

sns.distplot(data[data['Survived'] == 0]['Age'], hist=False, color="blue") 
sns.distplot(data[data['Survived'] == 1]['Age'], hist=False, color="orange")
plt.show()

pd.crosstab(data['Pclass'], data['Survived'])
sns.heatmap(pd.crosstab(data['Pclass'], data['Survived']))
plt.show()                        

sns.clustermap(pd.crosstab(data['Parch'], data['Survived']))
plt.show()
