# D-ratio
D ratio is a performance metric to analyse the efficiency of algorithms that predict asset return or asset prices.
This repository is linked to a paper still in preparation and should be understood with the paper. A larger repository for comparing algorithms predicting financial asset return or price will follow where D ratio will be applied.

D-ratio or 'Discriminant ratio' is a new performance ratio that measures the ability of an algorithm that predicts asset return (or asset prices) to efficiently improve the expected return and/or reduce the risk.
It compares the performance of the algorithm with the simple Buy & Hold strategy.

The risk of an investment is measured by the Value-at-Risk modified with the Cornish-Fisher expansion (CF-VaR). The CF-VaR is able to cope with non-Gaussian distributions whereas the standard VaR assumes normally distributed returns.
The return is the annualized return of the AI strategy compared to the annual return of the Buy & Hold strategy.

The D ratio can also be decomposed:
- to assess whether the added value of the algorithm is more linked to the improved expected return or to the risk reduction ability.
- to assess whether the algorithm efficiency is stable over time.

The merits of our proposed D ratio are numerous:

A.	D ratio better captures the risk of the asset as it is not limited by the assumption of a Gaussian distribution;

B.	D ratio is valid for all kinds of algorithms: ML, DL and RL, with regressions or classification;

C.	D ratio is time-insensitive: the efficiency of the algorithm can be compared over various periods of time. The stability of the algorithm can easily be verified by testing the     D ratio over the complete period versus two sub-periods. 
    As the stability of AI models is a constant point of attention, this feature is key for assessing the effectiveness of AI algorithms and to avoid non reproducible results. In    our numerical example, the D ratio would be computed on the entire 5 years period and on two sub-periods of 2.5 years.

D.	D ratio can be decomposed into a first sub-ratio D-return dedicated to the efficiency of the algorithm to improve the return and a second sub-ratio D-VaR that assess the       efficiency of the algorithm to reduce the risk.

E.	D ratio allows to compare the algorithms applied with a long only strategy or with short-selling strategies. It allows to measure easily and efficiently the impact of  transaction costs on the effectiveness of the algorithm to improve the risk/return of the investment strategy.

F.	D ratio allows to compare the efficiency of the algorithm with various assets, from the same asset class (here, stocks) or across various asset classes.
