#%% Lectura de datos
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow import keras

keras.utils.set_random_seed(2)
datos = loadtxt('diabetes.csv', delimiter=',')
print(datos)
#%% Definición de variables
X = datos[:,:-1]
Y = datos[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,
                                                    test_size=0.2,
                                                    random_state=0)

#%% Construcción de la red neuronal
# La función Dense crea adyacencias entre las neuronas de una capa a otra
# Adam (Adaptive Moment estimation optimizer) es una extensión del descenso de gradiente estocástico
rn = Sequential() # Red neuronal sequencial (por capas)
rn.add(Dense(12, input_shape=(8,), activation='relu'))
rn.add(Dense(8, activation='relu'))
rn.add(Dense(1, activation='sigmoid'))
rn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#%% Ajuste con los datos
# Epochs: Total de veces que se itera el dataset completo
# Batch size: Tamaño de una muestra pequeña que alimenta a la red neuronal
rn.fit(X_train, Y_train, epochs=150, batch_size=10)

# %% Evaluación del modelo
accuracy = rn.evaluate(X_test, Y_test)[1]
print(f'Accuracy: {round(accuracy*100,2)}%')

#%% Tuneo del número de capas
def red_neuronal(Xtrain, Ytrain, layers):
    best = 0
    total_layers = [i+1 for i in range(layers)]
    for l in total_layers:
        rn = Sequential()
        rn.add(Dense(12, input_shape=(8,), activation='relu'))
        for i in range(l):
            rn.add(Dense(8, activation='relu'))
        rn.add(Dense(1, activation='sigmoid'))
        rn.compile(loss='binary_crossentropy',
                   optimizer='adam', metrics=['accuracy'])
        rn.fit(Xtrain, Ytrain, epochs=150, batch_size=10)
        accuracy = rn.evaluate(X_test, Y_test)[1]
        if accuracy > best:
            modelo = rn
    return modelo

RN = red_neuronal(X_train,Y_train, layers = 10)

accuracy = RN.evaluate(X_test, Y_test)[1]
print(f'Accuracy: {round(accuracy*100,2)}%')

# %% Ejemplo de una regresión logística
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

students = [[0,0],[0.5,0],[1,0],[1.25,0],
           [1.5,0],[1.75,0],[2,0],[2.5,0],[3,0],
           [3.5,0],[1.75,1],[2.25,1],[2.75,1],
           [3.25,1],[4,1],[4.25,1],[4.5,1],[4.75,1],
           [5,1],[5.5,1]]

hours = [x for x,y in students]
grade = [y for x,y in students]

datos = pd.DataFrame()
datos["x"] = hours
datos["y"] = grade

model = smf.logit(formula = "y ~ x", data = datos).fit()

b0 = model.params[0]
b1 = model.params[1]

fig,ax = plt.subplots(1,1, figsize = (10,5))

sig = lambda x: 1/(1 + np.e**(-(b0 + b1*x)))
X = np.linspace(0,max(datos["x"]),100)

sns.set_theme(style = "whitegrid")
sns.lineplot(ax = ax, x = X, y = sig(X), linewidth = 2, color = "#F76A6A",
            label = "Probablidad de pasar")
sns.scatterplot(ax = ax, data = datos, x = "x", y = "y", color = "k", s = 60)

ax.set_xlabel("Horas de estudio")
ax.set_ylabel("Aprobó")
ax.set_title("Horas de estudio vs. Aprobar")
plt.show()