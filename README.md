# GSIH 2025

This repository contains three quantitative finance challenges implemented in Python:

1. Automated Market Making using inventory-aware quoting.
2. Exotic Basket Option Pricing using Monte Carlo simulation with implied volatility calibration.
3. Optimal Portfolio Hedging using sparse regression (LASSO).

All implementations are self-contained and designed to run independently.

---

## Table of Contents

1. Automated Market Making
2. Exotic Option Pricing via Monte Carlo
3. Optimal Hedging Strategy
4. Installation and Setup

---

# 1. Automated Market Making

## Overview

Implements an inventory-aware automated market making engine that dynamically quotes bid and ask prices using:

- Volatility estimation (EMA-smoothed log returns)
- Inventory-adjusted Avellaneda-Stoikov-style quoting
- Spread control based on market microstructure
- Hard inventory limits

The engine processes orderbook snapshots and public trade executions sequentially.

---

## Input Format

The script expects two CSV file paths via standard input:

```
<orderbook_file_path>
<trades_file_path>
```

Example:

```
orderbook_train.csv
public_trades_train.csv
```

### Required Orderbook Columns

- `timestamp`
- `bid_1_price`
- `ask_1_price`

### Required Trade Columns

- `timestamp`
- `price`
- `side` (buy/sell)

Only the first 3000 rows of each file are processed.

---

## Output

Generates:

```
submission.csv
```

Format:

```
timestamp,bid_price,ask_price
```

---

## Strategy Details

### 1. Volatility Estimation

Volatility is computed using:

- Log returns of mid prices
- Exponentially weighted moving average smoothing
- Price range normalization

```
σ_t = max(EMA_vol, range_vol, default_vol)
```

---

### 2. Quote Construction

Quotes follow an inventory-adjusted framework:

Quote center:

```
mid_price − inventory × risk_parameter × σ²
```

Spread width:

```
risk_parameter × σ² + (2 / risk_parameter) × log(1 + risk_parameter / liquidity_parameter)
```

Inventory skew:

```
inventory_impact = 0.5 × (inventory / max_inventory)
```

Final bid/ask are:

```
bid = center − spread + inventory_impact × spread
ask = center + spread + inventory_impact × spread
```

---

### 3. Risk Controls

- Maximum inventory: ±20 units
- Spread capped at 3× market spread
- Aggressive skew when inventory exceeds 80% of limit
- Tick-size rounding enforcement
- Quote validity delay (effective at t+1)

---

## Running

```
python submission.py
```

Then provide file paths via stdin.

---

# 2. Exotic Option Pricing using Monte Carlo Simulation

## Overview

Prices exotic basket options with up-and-out knock-out barriers using:

- Black-Scholes implied volatility calibration
- Correlated Geometric Brownian Motion
- Monte Carlo simulation
- Cholesky decomposition for correlation structure

---

## Market Setup

- Assets: DTC, DFC, DEC
- Spot price: 100
- Risk-free rate: 5%
- Correlation matrix:

```
[1.00  0.75  0.50]
[0.75  1.00  0.25]
[0.50  0.25  1.00]
```

---

## Calibration

Implied volatility is extracted using:

- Black-Scholes inversion via bisection
- Call prices across strikes [50, 75, 100, 125, 150]
- Maturities [1Y, 2Y, 5Y]

Volatility grids are cached for efficiency.

---

## Basket Definition

Basket price:

```
Basket(t) = (S₁(t) + S₂(t) + S₃(t)) / 3
```

Up-and-out barrier condition:

```
Option expires worthless if max(Basket(t)) ≥ Barrier
```

---

## Monte Carlo Simulation

Parameters:

- 6000 paths
- 81 steps per year
- Correlated Brownian motion via Cholesky
- Barrier monitored at every step

Drift:

```
(r − 0.5σ²)Δt
```

Diffusion:

```
σ√Δt Z
```

Final payoff:

Call:
```
max(Basket(T) − K, 0)
```

Put:
```
max(K − Basket(T), 0)
```

Discounted by:

```
e^{-rT}
```

---

## Output Format

Printed to stdout:

```
Id,Price
1,xx.xx
2,xx.xx
...
36,xx.xx
```

---

## Running

```
python submission.py
```

The script calibrates vol grids and prices all 36 basket options.

---

# 3. Optimal Hedging Strategy

## Overview

Constructs a sparse hedge portfolio using LASSO regression with cross-validation.

Objective:

Minimize residual exposure of portfolio PnL to stock returns.

---

## Input Format (Standard Input)

```
<portfolio_id> <pnl_1> <pnl_2> ... <pnl_T>
```

Example:

```
PORT_01 -2.5 1.8 -0.9 2.1
```

---

## Required Files

- `stocks_returns.csv`
- `stocks_metadata.csv`

Metadata must contain:

- Stock_Id
- Sector
- Rating
- Market_Cap
- Capital_Cost

---

## Methodology

### 1. Data Processing

- Returns normalized (percentage → decimal)
- Stocks sorted by:
  - Sector
  - Credit rating hierarchy
  - Market capitalization

### 2. Cost Adjustment

Returns scaled by inverse capital cost:

```
X = R × (1 / cost)
```

Target variable:

```
y = −portfolio_pnl
```

---

### 3. LASSO Regression

Model:

```
LassoCV(cv=5, fit_intercept=False)
```

- 5-fold cross-validation
- No intercept
- max_iter = 20000

Sparse coefficients produce hedge quantities.

Final positions:

```
positions = coefficients × cost_weights
```

Rounded to nearest integer.

---

## Output Format

```
<Stock_Id> <Quantity>
```

Only non-zero positions are printed.

---

## Running

```
echo "PORT_01 -2.5 1.8 -0.9 2.1" | python submission.py
```

---

# Installation

## Dependencies

```
pip install numpy pandas scipy scikit-learn
```

Python 3.8+

---

# Performance Characteristics

- Market Making: Processes 3000 timestamps efficiently
- Option Pricing: Monte Carlo simulation completes within seconds
- Hedging: LASSO solves in milliseconds for typical portfolio sizes

---

# Summary

This repository demonstrates:

- Inventory-aware microstructure trading models
- Monte Carlo pricing of path-dependent exotic derivatives
- Sparse optimization-based portfolio hedging
- Correlation modeling via Cholesky decomposition
- Volatility calibration using implied vol inversion

Each module is self-contained and reproducible.

