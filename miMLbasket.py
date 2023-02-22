import pandas as pd                                       
import seaborn as sns                                     
from sklearn.model_selection import train_test_split      
from sklearn.linear_model import LinearRegression         
from sklearn.tree import DecisionTreeRegressor            
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error            
import matplotlib.pyplot as plt                           
from gurobipy import Model, GRB, quicksum                 


# Cargamos el conjunto de datos boxscores
boxscores = pd.read_csv('https://raw.githubusercontent.com/Gurobi/modeling-examples/master/fantasy_basketball_1_2/boxscores_dataset.csv')     
boxscores = boxscores[(boxscores.playMin>=3) | (boxscores.playMin.isnull())]

# En general, los puntos de fantasía están conectados con características 
# relacionadas con el tiempo de juego o la eficiencia de cada jugador, y podemos 
# examinar algunas de estas relaciones a través de la visualización. A continuación
# se presentan algunos diagramas de dispersión:
fig, (FGA, FGM, FTM, Min) = plt.subplots(1, 4, figsize=(14,5))
fig.tight_layout()

FGA.scatter(boxscores['playFGA'], boxscores['FantasyPoints'], c='blue', alpha = .2)
FGM.scatter(boxscores['playFGM'], boxscores['FantasyPoints'], c='lightblue', alpha = .2)
FTM.scatter(boxscores['playFTM'], boxscores['FantasyPoints'], c='coral', alpha = .2)
Min.scatter(boxscores['playMin'], boxscores['FantasyPoints'], c='purple', alpha = .2)

FGA.set_xlabel('Field Goal Attempts')
FGM.set_xlabel('Field Goals Made')
FTM.set_xlabel('Free Throws Made')
Min.set_xlabel('Minutes Played')

FGA.set_ylabel('Fantasy Points');
plt.show()

# A continuación se muestra una gráfica de distribución de los verdaderos puntos 
# de fantasía de los jugadores para los juegos anteriores:
hplot = sns.histplot(boxscores['FantasyPoints'], color="blue", label="Fantasy Points", kde=True, stat="density", linewidth=0, bins=20)
hplot.set_xlabel("Fantasy Points", fontsize = 12)
hplot.set_ylabel("Density", fontsize = 12)
sns.set(rc={"figure.figsize":(14, 5)})
plt.show()

horizon=3
for column_name in ['playPTS','playAST','playTO','playSTL','playBLK','playTRB','playFGA','playFTA','play2P%','play3P%','playFT%','playMin','teamDayOff','FantasyPoints']:
    boxscores['moving' + column_name] = boxscores.groupby(['playDispNm'])[column_name].transform(lambda x: x.rolling(horizon, 1).mean().shift(1))                           
    
boxscores.dropna(subset = ["movingplayPTS"], inplace=True)

# Transformamos las variables categóricas en variables 0-1. Si el partido es en 
# casa o fuera y si el jugador comenzará o vendrá desde el banquillo: 
boxscores['dummyTeamLoc'] = pd.get_dummies(data=boxscores['teamLoc'],drop_first=True)    
boxscores['dummyplayStat'] = pd.get_dummies(data=boxscores['playStat'],drop_first=True)  

# Ahora que el conjunto de datos se ha actualizado, pasamos a pronosticar los 
# puntos de fantasía:
forecasting_data = boxscores[boxscores.gmDate != '2017-12-25']  

# División del conjunto de datos
X = forecasting_data[['movingplayAST','movingplayTO','movingplaySTL','movingplayBLK','movingplayTRB','movingplayFTA','movingplayFT%','dummyplayStat']]  
y = forecasting_data['FantasyPoints']  #target set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Modelo de Regresión Lineal ML
linear_regressor = LinearRegression()                                                         
linear_regressor.fit(X_train, y_train)               
                                         
# Modelo de Árbol de Decisión para la Regresión ML
decision_tree_regressor = DecisionTreeRegressor()
decision_tree_regressor.fit(X_train, y_train)

# Modelo de Bosques Aleatorios para la Regresión ML
random_forest_regressor = RandomForestRegressor()
random_forest_regressor.fit(X_train, y_train)

# Modelo de Vector Soporte para la Regresión ML
linear_svm = SVR()
linear_svm.fit(X_train, y_train)

# Ahora, para evaluar el rendimiento de los modelos, calculamos sus valores de 
# error cuadrático medio (MSE):
linear_regression_predictions = linear_regressor.predict(X_test)
linear_regression_mse = mean_squared_error(y_test, linear_regression_predictions)

decision_tree_regression_predictions = decision_tree_regressor.predict(X_test)
decision_tree_regression_mse = mean_squared_error(y_test,decision_tree_regression_predictions)

random_forest_regression_predictions = random_forest_regressor.predict(X_test)
random_forest_regression_mse = mean_squared_error(y_test,random_forest_regression_predictions)

linear_svm_predictions = linear_svm.predict(X_test)
linear_svm_mse = mean_squared_error(y_test,linear_svm_predictions)

results = {'Linear Regression':[linear_regression_mse], 'Decision Tree Regression':
        [decision_tree_regression_mse], 'Random Forest Regression':[random_forest_regression_mse], 
        'Redes neuronales':[linear_svm_mse]}
pd.options.display.max_columns = None
modeling_results = pd.DataFrame(data=results,index=['MSE'])
pd.options.display.max_columns = None
print(modeling_results)

fig, (LR, DTR, RFR, SVR) = plt.subplots(1, 4,figsize=(20,5));
fig.tight_layout()

LR.scatter(x = linear_regression_predictions, y = y_test - linear_regression_predictions,color='red',alpha=0.06)
DTR.scatter(x = decision_tree_regression_predictions, y = y_test - decision_tree_regression_predictions, color='green',alpha=0.06)
RFR.scatter(x = random_forest_regression_predictions, y = y_test - random_forest_regression_predictions, color='blue',alpha=0.06)
SVR.scatter(x = linear_svm_predictions, y = y_test - linear_svm_predictions, color='orange',alpha=0.06)

LR.set_xlabel('Puntos de Fantasía')
DTR.set_xlabel('Puntos de Fantasía')
RFR.set_xlabel('Puntos de Fantasía')
SVR.set_xlabel('Puntos de Fantasía')

LR.set_ylabel('Error Regresión Lineal')
DTR.set_ylabel('Error Árbol de Decisión')
RFR.set_ylabel('Error Bosque Aleatorio')
SVR.set_ylabel('Error Vector Soporte')


# OPTIMIZACIÓN

X = forecasting_data[['movingplayAST','movingplayTO','movingplaySTL','movingplayBLK','movingplayTRB','movingplayFTA','movingplayFT%','dummyplayStat']]
y = forecasting_data['FantasyPoints'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.04, random_state=0)

# Modelo de Regresión Lineal ML
linear_regressor_final = LinearRegression()                                                             
linear_regressor_final.fit(X_train, y_train)                                                        

# # Modelo de Árbol de Decisión para la Regresión ML
# decision_tree_regressor_final = DecisionTreeRegressor()
# decision_tree_regressor_final.fit(X_train, y_train)

# # Modelo de Bosques Aleatorios para la Regresión ML
# random_forest_regressor_final = RandomForestRegressor()
# random_forest_regressor_final.fit(X, y)

# # Modelo de Vector Soporte para la Regresión ML
# linear_svm_final = SVR()
# linear_svm_final.fit(X, y)

optimization_dataset = boxscores
optimization_dataset['PredictedFantasyPoints'] = linear_regressor_final.predict(boxscores[['movingplayAST','movingplayTO','movingplaySTL','movingplayBLK','movingplayTRB','movingplayFTA','movingplayFT%','dummyplayStat']]) 
# optimization_dataset['PredictedFantasyPoints'] = decision_tree_regressor_final.predict(boxscores[['movingplayAST','movingplayTO','movingplaySTL','movingplayBLK','movingplayTRB','movingplayFTA','movingplayFT%','dummyplayStat']]) 
# optimization_dataset['PredictedFantasyPoints'] = random_forest_regressor_final.predict(boxscores[['movingplayAST','movingplayTO','movingplaySTL','movingplayBLK','movingplayTRB','movingplayFTA','movingplayFT%','dummyplayStat']]) 
# optimization_dataset['PredictedFantasyPoints'] = linear_svm_final.predict(boxscores[['movingplayAST','movingplayTO','movingplaySTL','movingplayBLK','movingplayTRB','movingplayFTA','movingplayFT%','dummyplayStat']]) 


player_results = pd.read_csv('https://raw.githubusercontent.com/Gurobi/modeling-examples/master/fantasy_basketball_1_2/target_games.csv')
player_list = list(player_results['Player'].unique())
col = pd.DataFrame()

for player in player_list:    
    player_flag = player    
    optimization_data_per_player = optimization_dataset.loc[(optimization_dataset['playDispNm']==player)&(optimization_dataset['gmDate']=='2017-12-25')]
    col = col.append(optimization_data_per_player)

player_results['PredictedFantasyPoints'] = col['PredictedFantasyPoints'].values

indices = player_results.Player
points = dict(zip(indices, player_results.PredictedFantasyPoints))
salaries = dict(zip(indices, player_results.Salary))
S = 30000

m = Model();        

y = m.addVars(player_results.Player, vtype=GRB.BINARY, name="y")


m.setObjective(quicksum(points[i]*y[i] for i in indices), GRB.MAXIMIZE)

player_position_map = list(zip(player_results.Player, player_results.Pos))
for j in player_results.Pos:
    m.addConstr(quicksum([y[i] for i, pos in player_position_map if pos==j])==1)
    
m.addConstr(quicksum(salaries[i]*y[i] for i in indices) <= S, name="salary")       


m.optimize()  

results = pd.DataFrame()

for v in m.getVars():
    if v.x > 1e-6:
        results = results.append(player_results.iloc[v.index][['Player','Pos','PredictedFantasyPoints','Salary']])
        print(v.varName, v.x)

print('Total fantasy score: ', m.objVal)
print(results)




