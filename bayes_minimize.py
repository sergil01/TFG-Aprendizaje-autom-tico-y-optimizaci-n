import pandas as pd                                       #importing pandas
from sklearn.model_selection import train_test_split      #importing scikit-learn's function for data splitting
from sklearn.tree import DecisionTreeRegressor            #importing scikit-learn's decision tree regressor function
from sklearn.metrics import mean_squared_error            #importing scikit-learn's root mean squared error function for model evaluation
from skopt import gp_minimize

# Cargamos el conjunto de datos boxscores
boxscores = pd.read_csv('https://raw.githubusercontent.com/Gurobi/modeling-examples/master/fantasy_basketball_1_2/boxscores_dataset.csv')     
boxscores = boxscores[(boxscores.playMin>=3) | (boxscores.playMin.isnull())]

horizon=3
for column_name in ['playPTS','playAST','playTO','playSTL','playBLK','playTRB','playFGA','playFTA','play2P%','play3P%','playFT%','playMin','teamDayOff','FantasyPoints']:
    boxscores['moving' + column_name] = boxscores.groupby(['playDispNm'])[column_name].transform(lambda x: x.rolling(horizon, 1).mean().shift(1))                           #lagged moving average of numeric features
    
boxscores.dropna(subset = ["movingplayPTS"], inplace=True)

# Transformamos las variables categóricas en variables 0-1. Si el partido es en 
# casa o fuera y si el jugador comenzará o vendrá desde el banquillo: 
boxscores['dummyTeamLoc'] = pd.get_dummies(data=boxscores['teamLoc'],drop_first=True)    #1 si el partido se juega en casa, 0 en caso contrario
boxscores['dummyplayStat'] = pd.get_dummies(data=boxscores['playStat'],drop_first=True)  #1 si el jugador comienza jugando el partido, 0 en caso contrario

# Ahora que el conjunto de datos se ha actualizado, pasamos a pronosticar los 
# puntos de fantasía:
forecasting_data = boxscores[boxscores.gmDate != '2017-12-25']  #excluimos las observaciones del 25 de diciembre de 2017

# División del conjunto de datos
X = forecasting_data[['movingplayAST','movingplayTO','movingplaySTL','movingplayBLK','movingplayTRB','movingplayFTA','movingplayFT%','dummyplayStat']]  #select the features that will be used for model training
y = forecasting_data['FantasyPoints']  #target set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.04, random_state=0)
decision_tree_regressor = DecisionTreeRegressor()
decision_tree_regressor.fit(X_train, y_train)

def objective(params, X, y):
    decision_tree_regressor = DecisionTreeRegressor(max_depth=params)
    decision_tree_regressor.fit(X, y)
    decision_tree_regression_predictions = decision_tree_regressor.predict(X_test)
    score = mean_squared_error(y_test,decision_tree_regression_predictions)
    return score

def count_depths(children_left, children_right, node_id=0):
    if children_left[node_id] == children_right[node_id]:
        return 1
    else:
        left_depth = count_depths(children_left, children_right, children_left[node_id])
        right_depth = count_depths(children_left, children_right, children_right[node_id])
        return max(left_depth, right_depth) + 1
    
    
children_left = decision_tree_regressor.tree_.children_left
children_right = decision_tree_regressor.tree_.children_right

depth = count_depths(children_left, children_right)

bounds = [{'name': 'profundidad', 'type': 'discrete', 'domain': range(1, depth+1), 'integer': True}]
result = gp_minimize(lambda params: objective(params, X_train, y_train), bounds)
best_params = result.x