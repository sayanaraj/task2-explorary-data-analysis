import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

print(df.head())
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

print("\nSurvival Count:\n", df['Survived'].value_counts())
print("\nEmbarked Value Counts:\n", df['Embarked'].value_counts())
print("\nSex Distribution:\n", df['Sex'].value_counts())

plt.figure(figsize=(10, 4))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10, 4))
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survived")
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title("Fare vs Survived")
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title("Age vs Passenger Class")
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title("Survival Count by Gender")
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title("Survival Count by Class")
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title("Survival Count by Embarkation")
plt.show()

plt.figure(figsize=(10, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df[['Age', 'Fare', 'Survived', 'Pclass']], hue='Survived')
plt.suptitle("Pairplot of Age, Fare, Pclass vs Survived", y=1.02)
plt.show()

df.to_csv("titanic_eda_cleaned.csv", index=False)
print("✅ Cleaned data saved as 'titanic_eda_cleaned.csv'")
