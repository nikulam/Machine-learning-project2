import pandas as pd
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt

stats_df = pd.read_csv('curry.csv', delimiter=',')
#Filter preseason-games:
stats_df = stats_df[stats_df['Season_div'] != 'Pre']
#Filter outliers
stats_df = stats_df[stats_df['MIN'] > 15]
stats_df = stats_df[stats_df['MIN'] < 50]
stats_df = stats_df.reset_index(drop=True)

avg_3ptm = []
avg_pts = []
first_games = [0]
season_avgs = []

for i in range(1, stats_df.shape[0]):
    last_n_3ptm = []
    last_n_pts = []
    whole_season = []

    current_season = stats_df.iloc[i]['Season_year']
    
    #Get previous 9 rows or first i rows.
    prev_df = stats_df.iloc[max(0, i-9):i]
    for j, row in prev_df.iterrows():
        if current_season == row['Season_year']:
            last_n_3ptm.append(row['3PTM'])
            last_n_pts.append(row['PTS'])
    
    season_df = stats_df.iloc[first_games[-1]:i]
    for k, row in season_df.iterrows():
        if current_season == row['Season_year']:
            whole_season.append(row['PTS']) 
        
    if last_n_pts:
        avg_3ptm.append(stats.mean(last_n_3ptm))
        avg_pts.append(stats.mean(last_n_pts))
        season_avgs.append(stats.mean(whole_season))

    else:
        first_games.append(i)

    

new_stats = stats_df.drop(first_games)
new_stats.drop(new_stats.columns.difference(['OPP', 'Result', 'T Score', 'PTS']), 1, inplace=True)
new_stats['LAST_N_3PTM'] = avg_3ptm
new_stats['LAST_N_PTS'] = avg_pts
new_stats['SEASON_AVG'] = season_avgs

results = new_stats['Result'].map(dict(W=1, L=0))
new_stats['Result'] = results
new_stats = new_stats[['OPP', 'Result', 'T Score', 'LAST_N_3PTM', 'LAST_N_PTS', 'SEASON_AVG', 'PTS']]

#new_stats.to_csv('new_curry.csv', index=False)
