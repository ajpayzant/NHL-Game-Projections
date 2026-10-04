import time, pandas as pd, numpy as np
import config as C, features
t=time.time()
pg,tg=features.load_history()
shots=pd.read_parquet(C.LAKE/'shots_xg.parquet',columns=['game_id','date','season','opp','goalie','event','xg_on','target_net_empty'])
f=features.build(pg,tg,shots)
print('secs',time.time()-t, f.shape)
pd.set_option('display.width',250); pd.set_option('display.max_columns',60)
cols=['name','team','opp','given_line','given_pp','line','pp_unit','ev_share_4','slot_ev','aslot_ev','pp_share_4','slot_pp','sog_pm_ev','sh_pct','gf_pm_ev','ashare_ev','mate','opp_supp_ev','opp_supp_g','opp_gsax','rink','rest','ew_pp','opp_ew_sh']
print(f.loc[f.season.eq(2025)&f.name.astype(str).isin(['Connor McDavid','Nathan MacKinnon','Cale Makar','Ryan Lomberg'])].tail(6)[cols].T)
print(f[cols[7:]].describe().T.round(3))
f.to_parquet('data/features_cache.parquet')
