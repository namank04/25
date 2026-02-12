import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

def main():
    data = sys.stdin.read().strip().split()
    if len(data) < 2:
        return

    portfolio_id = data[0]
    pnl = np.array(data[1:], dtype=float)

    # Load returns and metadata
    returns_data = pd.read_csv('stocks_returns.csv', parse_dates=['Date'])
    merged_data = pd.read_csv('stocks_metadata.csv')

    tickers = [c for c in returns_data.columns if c != 'Date']
    meta = merged_data.copy()

    # Ensure filtering works
    meta = meta[meta['Stock_Id'].isin(tickers)].copy()

    # Custom sort keys
    rating_order = ['C', 'CCC-', 'CCC', 'CCC+', 'B-', 'B', 'B+', 'BB-', 'BB', 'BB+',
                    'BBB-', 'BBB', 'BBB+', 'A-', 'A', 'A+', 'AA-', 'AA', 'AA+', 'AAA']
    rating_order_map = {r: i for i, r in enumerate(rating_order)}
    meta['Rating_Sort'] = meta['Rating'].map(rating_order_map)

    cap_order_map = {'Large': 0, 'Medium': 1, 'Small': 2}
    meta['Cap_Sort'] = meta['Market_Cap'].map(cap_order_map)

    # Remove missing mapping rows
    meta = meta.dropna(subset=['Rating_Sort', 'Cap_Sort'])

    # Sort by sector, rating, and cap
    meta_sorted = meta.sort_values(by=['Sector', 'Rating_Sort', 'Cap_Sort'])

    sorted_tickers = meta_sorted['Stock_Id'].tolist()

    # Align returns
    R = returns_data[sorted_tickers].values / 100.0  # shape: (T, N)
    T, N = R.shape

    # Align capital cost
    costs = meta_sorted['Capital_Cost'].astype(float).values
    cost_weights = 1.0 / costs
    X = R * cost_weights[np.newaxis, :]
    y = -pnl

    model = LassoCV(cv=5, fit_intercept=False, max_iter=20000, random_state=42)
    model.fit(X, y)

    raw_coef = model.coef_
    positions = raw_coef * cost_weights
    qty = np.rint(positions).astype(int)

    for ticker, q in zip(sorted_tickers, qty):
        if q != 0:
            print(f"{ticker} {q}")

if __name__ == '__main__':
    main()

