# Log of experiments and results for the April 2025 project.

## 1. Imputer

- IterativeImputer: 12.6271
- SimpleImputer(mean): 12.5807
- SimpleImputer(median): 12.5814

## XGBoost

- Base: 12.6906
- Added Linear prediction of Episode_Length_minutes: 12.5868 (-0.1038)
- Five additional combined categorical features: 12.5105 (-0.1801)
