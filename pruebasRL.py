import pandas as pd                                       #importing pandas
from sklearn.model_selection import train_test_split      #importing scikit-learn's function for data splitting
from sklearn.linear_model import LinearRegression         #importing scikit-learn's linear regressor function
from sklearn.metrics import mean_squared_error            #importing scikit-learn's root mean squared error function for model evaluation
import matplotlib.pyplot as plt                           #importing matplotlib 

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
y = forecasting_data['FantasyPoints'] 

results = {}
k = 0.01

while k < 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=k, random_state=0)     
    
    # Modelo de Regresión Lineal ML
    linear_regressor = LinearRegression()                                                         
    linear_regressor.fit(X_train, y_train)                                                        
    
    # Ahora, para evaluar el rendimiento de los modelos, calculamos sus valores de 
    # error cuadrático medio (MSE):
    linear_regression_predictions = linear_regressor.predict(X_test)
    linear_regression_mse = mean_squared_error(y_test, linear_regression_predictions)
    
    results[k] = [linear_regression_mse]
    k = k + 0.01

min_key, min_value = min(results.items(), key=lambda x: x[1])
modeling_results = pd.DataFrame(data=results,index=['MSE'])
pd.options.display.max_columns = None
print(modeling_results)
print(min_key, min_value)

labels = list(results.keys())
values = list(results.values())

# Crear la gráfica de barras
plt.plot(labels, values)

# Añadir título y etiquetas a los ejes
plt.title("Variación del MSE")
plt.xlabel("% datos de prueba")
plt.ylabel("Error")

# Mostrar la gráfica
plt.show()
