{ Model: RandomForestRegressor
Targets: Protein Yield
Best Params: {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
Train CV R²: 0.79
Test R²: 0.81
MAE: 4.11
RMSE: 5.67 } , 
{ 
  Model: RandomForestRegressor  
  Target: Wheat Yield  
  Best Params: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}  
  Train CV R²: 0.772  
  Test R²: 0.698  
  MAE: 5.25  
  RMSE: 7.72  

model explains ~70% of the yield variance for wheat:
R² (0.772 train / 0.698 test)
}





