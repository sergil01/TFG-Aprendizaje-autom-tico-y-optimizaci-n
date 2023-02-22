import pandas as pd                                       
from sklearn.model_selection import train_test_split      
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error            

kernel = ['lineal','poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']

# Cargamos el conjunto de datos boxscores
boxscores = pd.read_csv('https://raw.githubusercontent.com/Gurobi/modeling-examples/master/fantasy_basketball_1_2/boxscores_dataset.csv')     
boxscores = boxscores[(boxscores.playMin>=3) | (boxscores.playMin.isnull())]

horizon=3
for column_name in ['playPTS','playAST','playTO','playSTL','playBLK','playTRB','playFGA','playFTA','play2P%','play3P%','playFT%','playMin','teamDayOff','FantasyPoints']:
    boxscores['moving' + column_name] = boxscores.groupby(['playDispNm'])[column_name].transform(lambda x: x.rolling(horizon, 1).mean().shift(1))                          
    
boxscores.dropna(subset = ["movingplayPTS"], inplace=True)

# Transformamos las variables categóricas en variables 0-1. Si el partido es en 
# casa o fuera y si el jugador comenzará o vendrá desde el banquillo: 
boxscores['dummyTeamLoc'] = pd.get_dummies(data=boxscores['teamLoc'],drop_first=True)    #1 si el partido se juega en casa, 0 en caso contrario
boxscores['dummyplayStat'] = pd.get_dummies(data=boxscores['playStat'],drop_first=True)  #1 si el jugador comienza jugando el partido, 0 en caso contrario

# Ahora que el conjunto de datos se ha actualizado, pasamos a pronosticar los 
# puntos de fantasía:
forecasting_data = boxscores[boxscores.gmDate != '2017-12-25']  #excluimos las observaciones del 25 de diciembre de 2017

# División del conjunto de datos
X = forecasting_data[['movingplayAST','movingplayTO','movingplaySTL','movingplayBLK','movingplayTRB','movingplayFTA','movingplayFT%','dummyplayStat']]  
y = forecasting_data['FantasyPoints'] 

k = 0.05
error = float('inf')
result = {}
while k < 1:
    for i in kernel:
        for j in C:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=k, random_state=0)     
            
            # Modelo de Regresión Lineal ML
            linear_svm = SVR(kernel='rbf', C=1.0)
            linear_svm.fit(X_train, y_train)                                                       
    
            # Ahora, para evaluar el rendimiento de los modelos, calculamos sus valores de 
            # error cuadrático medio (MSE):
            linear_svm_predictions = linear_svm.predict(X_test)
            linear_svm_mse = mean_squared_error(y_test, linear_svm_predictions)
            
            if linear_svm_mse < error:
                result['%'] = k
                result['kernel'] = i
                result['C'] = j
            k = k + 0.05

print(result)

