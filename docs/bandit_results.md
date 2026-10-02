# RetainAI — Offer Optimization Results (simulation)

At-risk customers: 1,303 | 50,000 simulated arrivals x 3 seeds

## Online learning

| policy                  |   net_revenue_per_customer |   lift_vs_no_offer |   total_regret_usd |   regret_per_customer_last_10k |
|:------------------------|---------------------------:|-------------------:|-------------------:|-------------------------------:|
| Always NO_OFFER         |                     473.44 |               0    |        1.03077e+06 |                          20.52 |
| Always LOYALTY_DISCOUNT |                     452.32 |             -21.12 |        2.09061e+06 |                          41.75 |
| Random eligible offer   |                     472.94 |              -0.5  |        1.01571e+06 |                          20.26 |
| Business rule           |                     485.39 |              11.95 |   424676           |                           8.5  |
| LinUCB                  |                     485.1  |              11.66 |   434450           |                           6.87 |
| Thompson sampling       |                     477.18 |               3.74 |   748630           |                          12.71 |
| Oracle                  |                     493.78 |              20.34 |        0           |                           0    |

## Final learned policy (noise-free)

|                             |   value_per_customer_usd |   agrees_with_oracle |   gap_to_oracle_usd |
|:----------------------------|-------------------------:|---------------------:|--------------------:|
| Always NO_OFFER             |                  472.175 |                0.029 |              20.53  |
| Business rule               |                  484.257 |                0.804 |               8.449 |
| LinUCB (learned)            |                  489.72  |                0.767 |               2.986 |
| Thompson sampling (learned) |                  475.303 |                0.276 |              17.402 |
| Oracle                      |                  492.705 |                1     |               0     |

## Simulator assumptions

|                         |   value |
|:------------------------|--------:|
| horizon_months          |   12    |
| causal_shrinkage        |    0.5  |
| accept_addon_trial      |    0.6  |
| accept_contract_new     |    0.25 |
| accept_contract_tenured |    0.45 |
| accept_discount         |    1    |
| addon_cost_per_month    |    8    |
| contract_discount_pct   |   10    |
| discount_months         |    6    |
