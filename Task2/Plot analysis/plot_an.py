import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('./train_data_final.csv')
fig = plt.figure(figsize=(16,9))
sns.barplot(x="security level of the community", y="total cost", data=df)
plt.xticks(fontsize=10)
plt.yticks(fontsize=20)
plt.show()
#plt.grid(True)
fig = plt.figure(figsize=(16,9))
plt.scatter(df['residence space'], df["total cost"])
plt.xticks(fontsize=10)
plt.yticks(fontsize=20)
#plt.title('',fontsize=20)
plt.xlabel('residence space',fontsize=20)
plt.ylabel('total cost ',fontsize=20)
plt.show()
