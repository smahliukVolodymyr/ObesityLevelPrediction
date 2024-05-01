outlier_columns = ['Age', 'Height','Weight',
                   'FCVC','NCP', 'CH2O','FAF','TUE']

cat_columns = ['Gender','CALC','FAVC','SCC','SMOKE','family_history_with_overweight',"CAEC","MTRANS","NObeyesdad"]

#### Define target and features columns

y_column = 'NObeyesdad' # target variable

X_columns = [ 'Weight','Gender','FCVC','family_history_with_overweight','NCP',"CAEC",'SCC','FAVC','Age','FAF','Height']