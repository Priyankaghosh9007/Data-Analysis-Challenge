import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df=pd.read_csv("Data Analysis Challenge Data Set.csv")

df_miss=df.replace("?",np.NAN)



df_change=df[df['normalized-losses']!='?']
change=df_change['normalized-losses'].astype(int).mean()
df['normalized-losses']=df['normalized-losses'].replace('?',change).astype(int)

df_change=df[df['bore']!='?']
change=df_change['bore'].astype(float).mean()
df['bore']=df['bore'].replace('?',change).astype(float)

df_change=df[df['stroke']!='?']
change=df_change['stroke'].astype(float).mean()
df['stroke']=df['stroke'].replace('?',change).astype(float)


df_change=df[df['price']!='?']
change=df_change['price'].astype(int).mean()
df['price']=df['price'].replace('?',change).astype(int)

df_change=df[df['horsepower']!='?']
change=df_change['horsepower'].astype(int).mean()
df['horsepower']=df['horsepower'].replace('?',change).astype(int)
                 
df_change=df[df['peak-rpm']!='?']
change=df_change['peak-rpm'].astype(int).mean()
df['peak-rpm']=df['peak-rpm'].replace('?',change).astype(int)

df['num-of-doors']=df['num-of-doors'].replace('?','four')

Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
IQR=Q3-Q1
print(IQR)

plt.rcParams['figure.figsize']=(19,7)
sns.boxplot(x="body-style", y="price", data=df)
plt.show()      


df=df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR))).any(axis=1)]





df[['engine-size','peak-rpm','curb-weight','horsepower','price']].hist(figsize=(10,8),bins=10,color='green')
plt.show()


plt.figure(1)
plt.subplot(221)
df['num-of-doors'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='red')
plt.title("Number of Door frequency diagram")
plt.ylabel('Number of Doors')
plt.xlabel('num-of-doors');

plt.subplot(222)
df['fuel-type'].value_counts(normalize= True).plot(figsize=(10,8),kind='bar',color='green')
plt.title("Number of Fuel Type frequency diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('fuel-type');

plt.subplot(223)
df['engine-type'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='yellow')
plt.title("Number of Engine Type frequency diagram")
plt.ylabel('Number of Engine Type')
plt.xlabel('engine-type');

plt.subplot(224)
df['body-style'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='blue')
plt.title("Number of Body Style frequency diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('body-style');
plt.tight_layout()
plt.show()


corr = df.corr()
plt.figure(figsize=(20,9))
sns.heatmap(corr, annot=True, fmt='.3f')
plt.show()


plt.rcParams['figure.figsize']=(23,20)
sns.boxplot(x="make", y="price", data=df, palette=sns.color_palette(['darkmagenta', 'teal','tomato','blue', 'yellow']))
plt.xticks(rotation=45)
plt.show()

sns.barplot(y='price',x='make', data=df)
plt.xticks(rotation=45)
plt.show()

plt.scatter(y=df['price'],x=df['engine-size'], color='teal')
plt.title("Scatter of Engine Size vs. Price")
plt.xlabel('Engine-size')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.show()
plt.rcParams['figure.figsize']=(19,7)
sns.boxplot(x="body-style", y="price", data=df)
plt.show()

sns.catplot(data=df, x="body-style", y="price", hue="aspiration" ,kind="point",markers=["o", "x"], linestyles=["--", "-."], palette=sns.color_palette(['red', 'orange'] ))
plt.show()     

plt.rcParams['figure.figsize']=(10,5)
sns.boxplot(x="drive-wheels", y="price", data=df,  palette=sns.color_palette(['darkmagenta', 'teal','tomato'])) 
plt.show()

sns.factorplot(data=df, x="engine-type", y="engine-size", col="body-style",row="fuel-type")
plt.show()

sns.catplot(data=df, x="num-of-cylinders", y="horsepower",kind="violin", palette=sns.color_palette(['darkmagenta', 'teal','tomato','blue', 'brown','green', 'yellow']))
plt.show()

sns.catplot(data=df, y="normalized-losses", x="symboling" , hue="body-style" ,kind="point", palette=sns.color_palette(['darkmagenta', 'teal','tomato','blue', 'yellow']))
plt.show()

sns.pairplot(df[["city-mpg", "highway-mpg","horsepower", "price","engine-size", "curb-weight", "fuel-type"]], hue="fuel-type", diag_kind="hist", palette=sns.color_palette(['red', 'yellow']))
plt.show()

sns.pairplot(df[["engine-size", "curb-weight", "price", "fuel-type"]], hue="fuel-type", diag_kind="hist", palette=sns.color_palette(['magenta', 'gold']))
plt.show()

sns.pairplot(df[["city-mpg","highway-mpg", "horsepower", "price", "fuel-type"]], hue="fuel-type", diag_kind="hist", palette=sns.color_palette(['orange', 'green']))
plt.show()
            


