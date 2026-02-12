import numpy as np
import pandas as pd
import io
import pickle
import os
from scipy.stats import norm

def bs_call_price(S,K,T,r,sigma):
    if sigma < 1e-8 or T< 1e-8:
        return max(S- K, 0)
    d1= (np.log(S / K)+ (r +0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S* norm.cdf(d1) - K* np.exp(-r* T) *norm.cdf(d2)

def bs_implied_vol_call(S, K, T, r, price):
    if price < max(S-K * np.exp(-r* T),0)+ 1e-8:
        return 1e-8
    low, high = 1e-8, 3.0
    for _ in range(30):
        mid = (low + high)/ 2
        p = bs_call_price(S,K, T,r,mid)
        if abs(p - price)< 1e-5:
            return mid
        if p> price:
            high = mid
        else:
            low= mid
    return mid

SPOT= 100.0
RF= 0.05
ASSETS= ['DTC','DFC','DEC']
STRIKES =[50, 75, 100, 125, 150]
MATURITIES =[1, 2, 5]  # in years

corr = np.array([
    [1.0,0.75,0.5],
    [0.75,1.0,0.25],
    [0.5,0.25, 1.0]])
chol= np.linalg.cholesky(corr)

calib_data = '''
CalibIdx,Stock,Type,Strike,Maturity,Price
1,DTC,Call,50,1y,52.44
2,DTC,Call,50,2y,54.77
3,DTC,Call,50,5y,61.23
4,DTC,Call,75,1y,28.97
5,DTC,Call,75,2y,33.04
6,DTC,Call,75,5y,43.47
7,DTC,Call,100,1y,10.45
8,DTC,Call,100,2y,16.13
9,DTC,Call,100,5y,29.14
10,DTC,Call,125,1y,2.32
11,DTC,Call,125,2y,6.54
12,DTC,Call,125,5y,18.82
13,DTC,Call,150,1y,0.36
14,DTC,Call,150,2y,2.34
15,DTC,Call,150,5y,11.89
16,DFC,Call,50,1y,52.45
17,DFC,Call,50,2y,54.9
18,DFC,Call,50,5y,61.87
19,DFC,Call,75,1y,29.11
20,DFC,Call,75,2y,33.34
21,DFC,Call,75,5y,43.99
22,DFC,Call,100,1y,10.45
23,DFC,Call,100,2y,16.13
24,DFC,Call,100,5y,29.14
25,DFC,Call,125,1y,2.8
26,DFC,Call,125,2y,7.39
27,DFC,Call,125,5y,20.15
28,DFC,Call,150,1y,1.26
29,DFC,Call,150,2y,4.94
30,DFC,Call,150,5y,17.46
31,DEC,Call,50,1y,52.44
32,DEC,Call,50,2y,54.8
33,DEC,Call,50,5y,61.42
34,DEC,Call,75,1y,29.08
35,DEC,Call,75,2y,33.28
36,DEC,Call,75,5y,43.88
37,DEC,Call,100,1y,10.45
38,DEC,Call,100,2y,16.13
39,DEC,Call,100,5y,29.14
40,DEC,Call,125,1y,1.96
41,DEC,Call,125,2y,5.87
42,DEC,Call,125,5y,17.74
43,DEC,Call,150,1y,0.16
44,DEC,Call,150,2y,1.49
45,DEC,Call,150,5y,9.7
'''
calib_df= pd.read_csv(io.StringIO(calib_data.strip()))

basket_option_data = '''
Id,Asset,KnockOut,Maturity,Strike,Type
1,Basket,150,2y,50,Call
2,Basket,175,2y,50,Call
3,Basket,200,2y,50,Call
4,Basket,150,5y,50,Call
5,Basket,175,5y,50,Call
6,Basket,200,5y,50,Call
7,Basket,150,2y,100,Call
8,Basket,175,2y,100,Call
9,Basket,200,2y,100,Call
10,Basket,150,5y,100,Call
11,Basket,175,5y,100,Call
12,Basket,200,5y,100,Call
13,Basket,150,2y,125,Call
14,Basket,175,2y,125,Call
15,Basket,200,2y,125,Call
16,Basket,150,5y,125,Call
17,Basket,175,5y,125,Call
18,Basket,200,5y,125,Call
19,Basket,150,2y,75,Put
20,Basket,175,2y,75,Put
21,Basket,200,2y,75,Put
22,Basket,150,5y,75,Put
23,Basket,175,5y,75,Put
24,Basket,200,5y,75,Put
25,Basket,150,2y,100,Put
26,Basket,175,2y,100,Put
27,Basket,200,2y,100,Put
28,Basket,150,5y,100,Put
29,Basket,175,5y,100,Put
30,Basket,200,5y,100,Put
31,Basket,150,2y,125,Put
32,Basket,175,2y,125,Put
33,Basket,200,2y,125,Put
34,Basket,150,5y,125,Put
35,Basket,175,5y,125,Put
36,Basket,200,5y,125,Put
'''
basket_df =pd.read_csv(io.StringIO(basket_option_data.strip()))

def calibrate_lv_grids(force_recalibrate=False):
    cache_file= 'lv_grids_cache.pkl'
    if os.path.exists(cache_file) and not force_recalibrate:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    lv_grids ={}
    for asset in ASSETS:
        lv_grid = np.zeros((len(STRIKES),len(MATURITIES)))
        for i, K in enumerate(STRIKES):
            for j, T in enumerate(MATURITIES):
                price = calib_df[
                    (calib_df['Stock'] == asset) &(calib_df['Strike'] == K) &
                    (calib_df['Maturity'] == f"{T}y")]['Price'].values[0]
                lv_grid[i, j] = bs_implied_vol_call(SPOT, K, T, RF, price)
        lv_grids[asset] = lv_grid
    with open(cache_file,'wb') as f:
        pickle.dump(lv_grids, f)
    return lv_grids

def get_lv(asset, S,T, lv_grids):
    strike_idx = np.argmin([abs(S- k) for k in STRIKES])
    maturity_idx = np.argmin([abs(T -t) for t in MATURITIES])
    return lv_grids[asset][strike_idx, maturity_idx]


def price_basket_knockout(lv_grids, strike, T, rf, barrier, opt_type,n_paths=4000, n_steps_per_year=10):
    n_steps = int(T * n_steps_per_year)
    dt = T / n_steps
    n_assets = 3
    S = np.full((n_paths, n_assets),SPOT)
    alive = np.ones(n_paths, dtype=bool)
    max_basket = np.full(n_paths,SPOT)
    # Precompute sigmas for all assets
    sigmas = np.array([get_lv(asset, SPOT,T, lv_grids) for asset in ASSETS])
    drift = (rf - 0.5 * sigmas ** 2) * dt
    vol = sigmas * np.sqrt(dt)
    for _ in range(n_steps):
        z = np.random.normal(size=(n_paths, n_assets))
        z_corr = z @ chol.T
        S *= np.exp(drift + vol * z_corr)
        basket = S.mean(axis=1)
        max_basket = np.maximum(max_basket, basket)
        alive &= (max_basket < barrier)
        if not alive.any():  # All knocked out,stop hten 
            break
    basket_T = S.mean(axis=1)
    if opt_type.lower()== 'call':
        payoff = np.maximum(basket_T - strike, 0)
    else:
        payoff = np.maximum(strike - basket_T, 0)
    payoff[~alive]= 0.0
    price =np.exp(-rf * T) * np.mean(payoff)
    return max(price,0.0)


def main (force_recalibrate= False):
    lv_grids =calibrate_lv_grids(force_recalibrate= force_recalibrate)
    results = []
    for _, row in basket_df.iterrows():
        T =int(str(row['Maturity']).replace('y', ''))
        strike= float(row['Strike'])
        barrier =float(row['KnockOut'])
        opt_type= row['Type']
        price = price_basket_knockout(
            lv_grids,strike,T,RF, barrier,opt_type,
            n_paths=6000, n_steps_per_year=81  # Fast, accurate
        )
        results.append({'Id': row['Id'],'Price': round(price, 2)})
    print('Id,Price')
    for r in results:
        print(f"{r['Id']},{r['Price']}")

if __name__ == '__main__':
    main(force_recalibrate=False)