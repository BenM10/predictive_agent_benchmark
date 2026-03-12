# Experiment Notes - Task 2: Exploratory Data Analysis

## Observations
- **Target Distribution**: Satisfaction is relatively balanced, but there's a slight majority of "neutral or dissatisfied" passengers.
- **Key Features**: 
    - `Class`: Business class passengers show significantly higher satisfaction.
    - `Type of Travel`: Business travel is strongly associated with higher satisfaction compared to personal travel.
    - `Online boarding` and `Inflight wifi service`: These seem to be strong predictors of satisfaction.
- **Missing Data**: Confirmed 393 missing values in `Arrival Delay in Minutes`, which is highly correlated with `Departure Delay in Minutes`.

## Decisions
- Implemented `save_and_show` to automate figure saving in `figures/antigravity/`.
- Generated 7 visualizations covering target distribution, demographic splits, categorical features, and correlations.
- Used Seaborn for aesthetically pleasing and informative plots.
