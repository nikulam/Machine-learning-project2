import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

new_df = pd.read_csv('new_curry.csv', delimiter=',')
stats_df = pd.read_csv('curry.csv', delimiter=',')
career = pd.read_csv('career.csv', delimiter=',')

pca = PCA(n_components=1)
x_pca = list(pca.fit_transform(new_df[['LAST_N_3PTM', 'LAST_N_PTS', 'SEASON_AVG']]))

plt.scatter(x_pca, new_df['PTS'], color='green')
plt.xlabel('1st principal component of features')
plt.ylabel('PTS')
plt.show()

'''
plt.hist(stats_df['MIN'], bins=100, color='orange')
plt.xlabel('minutes played')
plt.ylabel('number of games')
plt.title('Number of games by minutes played')
plt.show()
'''

'''
sns.heatmap(new_df.corr(), cmap='GnBu')
plt.title('Correlation between features')
plt.show()
'''

plt.plot(career['3P'])
plt.plot(career['PTS'] / 3)
plt.show()