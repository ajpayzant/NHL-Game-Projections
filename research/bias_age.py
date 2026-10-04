import pandas as pd, numpy as np
b=pd.read_parquet('data/backtest/backtest_last.parquet'); b=b[(b.game_type==2)&(b.gp_16>0)]
bio=pd.read_parquet('data/lake/bio.parquet'); b=b.merge(bio[['player','birth']],on='player')
b['age']=(b.date-pd.to_datetime(b.birth)).dt.days/365.25; b['ab']=pd.cut(b.age,[17,21,23,25,27,29,31,33,45])
pg=pd.read_parquet('data/lake/player_games.parquet'); ls=pg[pg.game_type==2].groupby(['player','season']).agg(n=('points','size'),lp=('points','mean')).reset_index(); ls['season']+=1
b=b.merge(ls,on=['player','season'],how='left'); b['star']=b.lp.ge(0.8)&b.n.ge(20)
f=lambda s: pd.Series({'PTS':s.points.sum()/s.pts_hat.sum(),'A':s.assists.sum()/s.a_hat.sum(),'G':s.goals.sum()/s.g_hat.sum(),'SOG':s.sog.sum()/s.sog_hat.sum(),'n':len(s)})
print('ALL by age (actual/projected)'); print(b.groupby('ab',observed=True).apply(f,include_groups=False).round(3))
print('STARS by age'); print(b[b.star].groupby('ab',observed=True).apply(f,include_groups=False).round(3))
print('by season, under-28 vs 28+'); print(b.assign(y=b.age<28).groupby(['season','y']).apply(f,include_groups=False).round(3))
