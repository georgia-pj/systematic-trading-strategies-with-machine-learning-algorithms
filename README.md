This year's coursework asks you to build a metamodel on top of a primary trading signal that we provide for 11 instruments across three asset classes. The metamodel's job is to predict, for each primary signal, the probability that following it would be profitable under a triple-barrier exit rule.

The Universe
You are provided with the primary model's daily signals (-1, 0, +1) for the following 11 instruments.

Equity Index Futures
Ticker	Index
ES1S	S&P 500
NQ1S	Nasdaq 100
FESX1S	Euro Stoxx 50
Energy
Ticker	Commodity
CL1S	WTI Crude Oil
HO1S	Heating Oil
RB1S	RBOB Gasoline
NG1S	Natural Gas
Metals
Ticker	Metal
GC1S	Gold
SI1S	Silver
HG1S	Copper
PL1S	Platinum
You are required to cover at least one full asset class. Covering more (up to all 11 instruments) is optional.

Task Description
Build a metamodel for each instrument you cover. The metamodel takes the primary signal plus your features and outputs a probability in [0, 1] that the bet is worth taking.

The pipeline is:

Feature engineering from OHLCV (and anything else you can derive)
Labeling via the triple-barrier method, as taught in the course
Training and comparing several ML models with hyperparameter tuning
Feature importance analysis at the cluster level
Evaluation on a clean out-of-sample period
(Optional) Strategy construction on top of the metamodel probabilities
