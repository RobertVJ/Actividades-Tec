"""
Implementación de una técnica de aprendizaje máquina sin el uso de un framework

Autor: Roberto Valdez Jasso

Matricula: A01746863

Fecha de Inicio 30/08/2022

Fecha de Finalizacion: 2/09/2022

Entregable: Implementación de una técnica de aprendizaje máquina sin el uso de un framework.

1. Crea un repositorio de GitHub para este proyecto.


2. Programa uno de los algoritmos vistos en el módulo (o que tu profesor de módulo autorice) sin usar ninguna biblioteca o framework de aprendizaje máquina, ni de estadística avanzada. Lo que se busca es que implementes
manualmente el algoritmo, no que importes un algoritmo ya implementado.

3. Prueba tu implementación con un set de datos y realiza algunas predicciones. Las predicciones las puedes correr en consola o las puedes implementar con una interfaz gráfica apoyándote en los visto en otros módulos.

4. Tu implementación debe de poder correr por separado solamente con un compilador, no debe de depender de un IDE o de un “notebook”. Por ejemplo, si programas en Python, tu implementación final se espera que esté en un archivo .py no en un Jupyter Notebook.

5. Después de la entrega intermedia se te darán correcciones que puedes incluir en tu entrega final.

"""
#---------------------------------------------------------------------#

"""
Modelo  de Inteligencia Artificial Elegido

El Modelo Elegido para esta actividad es: Regresion Logistica.

Nota:
En esta actividad usare  un codigo que ya realice en cursos anteriores realizados en Linkedin con la finalidad de 
ver y comparar el porcentaje de acercamiento del realizado con librerias como tambien los valores pruebas regresado 
con una semilla random diferente al anterior.

"""
#---------------------------------------------------------------------#
# Librerias base para generar la la regresion Logistica
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  pylab import  rcParams
import  seaborn as sb
from  sklearn.model_selection import train_test_split
from  sklearn.metrics import  classification_report
from  sklearn.preprocessing import  LabelEncoder
from  sklearn.preprocessing import  OneHotEncoder
#---------------------------------------------------------------------#


#  LOGISTIC REGRESSION INITIAL CONCEPT AND PREPARATION

# Preparando los parametros de graficacion
# Dimensiones
rcParams['figure.figsize'] = 10, 8
# Estilo de grafico
sb.set_style('whitegrid')

#Regresion Logistica con un dataset titanico

# primero importamos los datos

address = r'C:\Users\rober\PycharmProjects\IATEC2022PART1\DATA\titanic-training-data.csv'
# creamos el dataframe
titanic_training= pd.read_csv(address)
titanic_training.columns = ['PassagerID','Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
print(f'Dataframe con valores del cabeza : \n{titanic_training.head()}') # imprimos los valores del dataframe
print("/------------------------------------/")
#Revisamos la informacion del dataset titanico
print(f'{titanic_training.info()}') # imprimos la informacion
print("/------------------------------------/")
"""
Descripcion de las variables

Survived: Survival (0 = NO, 1= YES) (variable binaria)
pClass = Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
Name = Name
Sex = Sex
Age = Age
SibSp = Number of Siblings/Spouses Aboard
Parch = Number of Parents/Chindren Aboard
Ticket = Number of Tickets
Fare = Passanger Fare (British Pound)
Cabin = Cabin
Embarked = Port of Embarkation ( C = Cherbour ,France;
                                 Q = Queenstown, UK;
                                 S = Southampton, Cobh- Ireland;                                            
                                ) variable categorica
"""


# DATA PREPARATON

# para preparar los datos primero necetamos checar si la variable tarjet es binaria
# Queremos checar, probar y predecir la sobrevencia , pero primero hay que checar si es binario
sb.countplot(x = "Survived", data = titanic_training, palette = 'hls')
plt.show() # vemos la grafica de sobreviencia denotando que si es binaria la variable

# Checamos por valores perdidos
# Siempre hay que hacerlo (en este caso si hay valores perdidos , lo podemos checar desde la info del dataset)
s = titanic_training.isnull()
print(f'El valor del dataset de valores perdidos es: \n{s}')
print("/------------------------------------/")
sum = titanic_training.isnull().sum()
print(f'Los valores perdidos son : \n{sum}') # vemos que en cabin (cabina le faltan todas) y en age (faltan algunos) faltan valores
print("/------------------------------------/")
# Otro metodo es:
print(f'El valor del dataset de valores perdidos es: \n{titanic_training.describe()}')
print("/------------------------------------/")

"""
Taking care of missing values
    Dropping missing values
    So let's just go ahead and drop all the variables that aren't relevant for predicting survival. We should at least keep the following:
    Survived - This variable is obviously relevant.
    Pclass - Does a passenger's class on the boat affect their survivability?
    Sex - Could a passenger's gender impact their survival rate?,
    Age - Does a person's age impact their survival rate?
    SibSp - Does the number of relatives on the boat (that are siblings or a spouse) affect a person survivability? Probability
    Parch - Does the number of relatives on the boat (that are children or parents) affect a person survivability? Probability
    Fare - Does the fare a person paid effect his survivability? Maybe - let's keep it.
    Embarked - Does a person's point of embarkation matter? It depends on how the boat was filled... Let's keep it

    What about a person's name, ticket number, and passenger ID number? They're irrelavant for predicting survivability.
    And as you recall, the cabin variable is almost all missing values, so we can just drop all of these.
"""

# Generamos un nuevo dataframe con la elimanacion de los campos anteriores
titanic_data = titanic_training.drop(['Name', 'Ticket', 'Cabin'], axis= 1) # tumbamos las columnas
print(f'El nuevo dataset de valores: \n{titanic_data.head()}') # dataset limpio con datos relevantes
print("/------------------------------------/")

# TREATING MISSING VALUES

# tratando con valores desaparecidos
# Agregamos los valores desaparecidos con aproximacion

# Primero checamos la distribuccion de los datos entre Parch y Age
sb.boxplot(x = "Parch",y= "Age", data = titanic_data, palette = 'hls')
plt.show() # podemos denotar que hay relaciones entre ambas variables, es decir, podemos ver  que puede haber dos relativos por bote
           # (mientras mas edad mauor cantidad hasta llegar poren encima de 40 años)

# Aproximacion de las edades de los pasajeros en base al numero de padres o hijos que hay en el bote
# generamos un dataframe para la aproximacion

Parch_groups = titanic_data.groupby(titanic_data['Parch'])
print(Parch_groups.mean())  # Sacamos la media de checar las edades
                            # por categoria tenemos la mediade las edades
                            # para las personas que tenga cero hijos o padres en el barco la edad promedio es de 32
                            # para las personas que tenga un hijo o padre en el barco la edad promedio es de 24
                            # para las personas que tenga dos hijos o padres en el barco la edad promedio es de 17
                            # para las personas que tenga tres hijos o padres en el barco la edad promedio es de 33
                            # para las personas que tenga cuatro hijos o padres en el barco la edad promedio es de 44
                            # para las personas que tenga cinco hijos o padres en el barco la edad promedio es de 39
                            # para las personas que tenga seis hijos o padres en el barco la edad promedio es de 43
print("/------------------------------------/")

# con lo anterior, agregamos los nuevos valores con una funcion

def age_aprox(cols):
    Age = cols[0]
    Parch = cols[1]
    # Checamos si  los valores esta vacios o no existen
    if pd.isnull(Age):
        # Si es asi usamos la aproximacion de grupos de hijos o padre por
        # media de edades de los pasajeros en el barco como esta arriba
        if Parch == 0:
            return 32
        elif Parch == 1:
            return  24
        elif Parch == 2:
            return  17
        elif Parch == 3:
            return  33
        elif Parch == 4:
            return  44
        else:
            return  39 # 5 hijos o padres
    else:
        return Age

# Agregamos los datos por aproximacion
titanic_data['Age'] = titanic_data[['Age', 'Parch']].apply(age_aprox, axis = 1)

# Checamos que no haya vacios (cosa ya no habra)
sum = titanic_data.isnull().sum()
print(f'Los valores perdidos son : \n{sum}') # no faltan valores en edades, pero faltan dos en embarked
print("/------------------------------------/")

# tumbamos las filas pertenecientes a eso dos valores perdidos
titanic_data.dropna(inplace = True)
titanic_data.reset_index(inplace = True, drop= True)
#Revisamos la informacion del dataset
print(f'{titanic_data.info()}') # imprimos la informacion
print("/------------------------------------/")


# RE-ENCODE VARIABLES

# Covertiendo variables categoricas a indicadores dummy

# vamos a reformatear la variable sex y embarked a variables numericas

# Hacemos esto para que funcionen dentro del modelo

label_encoder = LabelEncoder() # llamamos al encoder
# convertimos el genero
gender_Cat = titanic_data['Sex']
gender_Encoder = label_encoder.fit_transform(gender_Cat) # encoder la seccion sex
print(f'Resultado del encoder con variable sex: \n{gender_Encoder[0:5]}') # no sabemos que signifca 1 o 0
print("/------------------------------------/")

# para saberlo hacemos lo siguiente
print(f'Tabla Dataframe \n{titanic_data.head()}') # 1 = male
                                                  # 0 = female
print("/------------------------------------/")
# creando un gender Dataframe
gender_DF = pd.DataFrame(gender_Encoder, columns= ['male_gender'])
print(f'Nuevo Dataframe Gender \n{gender_DF.head()}')
print("/------------------------------------/")


# Ahora trabajaremos embarked
embarked_Cat = titanic_data['Embarked']
embarked_encoder=  label_encoder.fit_transform(embarked_Cat)
print(f'Resultado del encoder con variable sex: \n{embarked_encoder[0:100]}') # nos dio una variable categorica multi-nominal y necesitamos una variable binaria
print("/------------------------------------/") # indicadores nuevos dummy

# hacemos lo siguiente para conseguir una variable binaria de embarked
# necesitamo one hot encoder
binary_encoder = OneHotEncoder( categories= 'auto')
embarked_1hot = binary_encoder.fit_transform(embarked_encoder.reshape(-1,1)) # genera un array de una sola columna pero la queremos en matrix
embarked_1hot_mat = embarked_1hot.toarray() # para matrix
embarked_DF  = pd.DataFrame(embarked_1hot_mat, columns= ['C', 'Q', 'S']) #de la matrix a dataframe, las columnas representas en donde embarcaron
print(f'Nuevo Dataframe Embarked \n{embarked_DF.head()}')
print("/------------------------------------/") # indicadores nuevos dummy

# tiramos para abajo  las columnas ya no necesarias de Titanic data

titanic_data.drop(['Sex', 'Embarked'], axis = 1, inplace = True)
print(f'Nuevo arreglo de Titanic Dataframe \n{titanic_data.head()}')
print("/------------------------------------/")

# ahora concatenamos los nuevos indicadores a la tabla original
titanic_data_dmy = pd.concat([titanic_data,gender_DF,embarked_DF], axis= 1, verify_integrity= True).astype(float)
print(f'Nuevo arreglo de Titanic Dataframe con datos concatenados \n{titanic_data_dmy.head()}')
print("/------------------------------------/")


# VALIDATING DATASET

# Checando por independecias entre atributos

plt.figure()
sb.heatmap(titanic_data_dmy.corr()) # corelacionamos los datos
plt.show() # Lo que nos dice la grafica que esu si tenemos corelacion cerca a uno o uno negativos, significa que
           # se obtuvo una fuerte relacion lineal entre el par de variable
           # regresion logistica asume que los atributos debe ser independiente del uno con el otro
           # lo cual no podemos tener esto

# Fare y Plass no son independiente de uno y del otro vamos tumbarlas
titanic_data_dmy.drop(['Fare', 'Pclass'], axis= 1, inplace= True)
print(f'Nuevo arreglo de Titanic Dataframe \n{titanic_data_dmy.head()}') # datafame mas limpio
print("/------------------------------------/")


# Checmos si el dataset con todos los arreglos tiene un tamaño suficiente para hacer la regresion logistica

# primero hay que considera  cuando variables predicitivas tenemos
# se realiza por la regla de Thumb : realizar almenos 50 observaciones por variable predictiva (para asegurar valores de canfianza)
print(f'Todos los valores menos el valor Survived que es el que vamos a checar\nson valores predictorios:\n{titanic_data_dmy.head()}') # datafame
print("/------------------------------------/")

# Checamos si tenemos los datos suficiente
print(f'Cantidad de observaciones por valor: {titanic_data_dmy.info()}') # 889 datos por valor, lo cual mas que suficiente parala regresion logistica
print("/------------------------------------/")


# MODEL DEPLOYMENT

# Rompemos el dataframe para  el set entrenamiento (4/5 de datos del dataframe) y  set pruebas (pass set) (1/5 de datos del dataframe)
# y quitamos la variable Survived qque es la que queremos checar
X_train, X_test, Y_train, Y_test = train_test_split(titanic_data_dmy.drop(['Survived'], axis= 1), # Valores en X
                                                    titanic_data_dmy['Survived'], test_size = 0.2, # Valores en Y
                                                    random_state= 200) # seed de random para tener los mismos resultados
#checando los resultados
print(f'Valores de entrenamiento en X :\n{X_train.shape} \nvalores en Y:\n {Y_train}') # En y tenemos una ifla con 711 datos y en
print("/------------------------------------/")

#checando los resultados
print(f'Valores de entrenamiento predictores en X :\n{X_train[0:5]}')  #Checamos por los primeros 5 valores que usaramos como predictores
print("/------------------------------------/")


# Desplegando y evaluando el modelo de Regresion logistica

#------------------------------------------------------------------------------#
#                                PRIMERA PRUEBA                                #
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#                           SECCION SIN USO LIBRERIA                           #
#------------------------------------------------------------------------------#
# Regresion Logistica a "mano"

# Clase Regresion Logistica

class LogitRegression():
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate  # ratio de aprendizaje
        self.epochs = epochs  # interaciones del modelo

    # Funcion fit
    # Nos apoya generando el modelo para entrenar la regresion logistica
    # como tambien realizarla.
    def fit(self, X, Y):
        # Variables iniciales
        self.m, self.n = X.shape  # toman en tamaño (shape) de los datos de entrenamiento en
        # Estos deben ser de manera obligatoria datos de entrenamiento
        # y no de los de preuba, en caso contrario no funcionara el
        # modelo.
        # Inicializacion de los pesos
        self.W = np.zeros(self.n)  # peso inicial
        self.b = 0  # Bias inicial
        self.X = X  # Datos de entrenamiento en X  (vector)
        self.Y = Y  # Datos de entrenamiento en Y

        # Ciclo for para generar el aprendizaje
        # del gradiente descendiente.
        # se genera por las epocas decidas por el usuario
        for i in range(self.epochs):
            self.calculate_weights()  # Caluclo constante de los pesos
        return self

    # Funcion calculate weights
    # Funcion que nos apoya en generar los pesos
    # al momento de calcular el gradiente descendiente
    def calculate_weights(self):
        A = 1 / (1 + np.exp(- (self.X.dot(self.W) + self.b)))  # Alfa (aprendizaje)
        # Se calculan el gradiente desencidiente
        tmp = (A - self.Y.T)
        tmp = np.reshape(tmp, self.m)
        dW = np.dot(self.X.T, tmp) / self.m
        db = np.sum(tmp) / self.m
        # se actualizan los nuevos pesos con el ratio de aprendizaje
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return self  # regresamos la carga nueva de pesos

    # Funcion Predict
    # Genera la prediccion del modelo, en base a la formula
    # de funcion hipotetica.
    def predict(self, X):
        Z = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))  # sigmoide
        Y = np.where(Z > 0.5, 1, 0)
        return Y  # regresamos la prediccion


# Llamando al modelo
# Model training
#model = LogitRegression(learning_rate = 0.001, epochs = 100000)  # Si Funciona pero necesita muchas MUCHAS Interaciones como tambien el
#model.fit(X_train, Y_train)  # tarda su rato dependiendo las epocas y el ratio de apredizaje que se le ponga

# Prediction on test set
#Y_pred = model.predict(X_test)

#print(f'Prediccion Modelo realizado de prueba 1:\n {Y_pred}')  # Checamos por los primeros 5 valores que usaramos como predictores
print("/------------------------------------/")

# Reporte de Clasificacion  sin cross validation del modelo generado
#print(f'Reporte de clasificacion sin crossvalidation  de prueba 1:\n {classification_report(Y_test, Y_pred)}') # vemos 43 de presicion de predicion que nada bueno pero puede mejorar
                                                                                                   # modificando el learning ratio y las epocas realizadas
print("/------------------------------------/")

#------------------------------------------------------------------------------#
#                                SEGUNDA PRUEBA                                #
#------------------------------------------------------------------------------#

# MODEL DEPLOYMENT

# Preidicion de quien es hombre y quien es mujer
# Rompemos el dataframe para  el set entrenamiento (4/5 de datos del dataframe) y  set pruebas (pass set) (1/5 de datos del dataframe)
# y quitamos la variable Survived qque es la que queremos checar
X_train, X_test, Y_train, Y_test = train_test_split(titanic_data_dmy.drop(['male_gender'], axis= 1), # Valores en X
                                                    titanic_data_dmy['male_gender'], test_size = 0.2, # Valores en Y
                                                    random_state= 200) # seed de random para tener los mismos resultados

#------------------------------------------------------------------------------#
#                           SECCION SIN USO LIBRERIA                           #
#------------------------------------------------------------------------------#

# Llamando al modelo
# Model training
#model = LogitRegression(learning_rate = 0.01, epochs = 999)  # Si Funciona pero necesita muchas MUCHAS Interaciones como tambien el
#model.fit(X_train, Y_train)  # tarda su rato dependiendo las epocas y el ratio de apredizaje que se le ponga

# Prediction on test set
#Y_pred = model.predict(X_test)

#print(f'Prediccion Modelo realizado:\n {Y_pred}')  # Checamos por los primeros 5 valores que usaramos como predictores
print("/------------------------------------/")

# Reporte de Clasificacion  sin cross validation del modelo generado
#print(f'Reporte de clasificacion sin crossvalidation de la segunda prueba :\n {classification_report(Y_test, Y_pred)}')
print("/------------------------------------/")

#------------------------------------------------------------------------------#
#                                TERCERA PRUEBA                                #
#------------------------------------------------------------------------------#

# MODEL DEPLOYMENT

# Preidicion de CUANTOS familiares tienen en el barco
# Rompemos el dataframe para  el set entrenamiento (4/5 de datos del dataframe) y  set pruebas (pass set) (1/5 de datos del dataframe)
# y quitamos la variable Survived qque es la que queremos checar
X_train, X_test, Y_train, Y_test = train_test_split(titanic_data_dmy.drop(['SibSp'], axis= 1), # Valores en X
                                                    titanic_data_dmy['SibSp'], test_size = 0.2, # Valores en Y
                                                    random_state= 200) # seed de random para tener los mismos resultados


#------------------------------------------------------------------------------#
#                           SECCION SIN USO LIBRERIA                           #
#------------------------------------------------------------------------------#
# Llamando al modelo
# Model training
model = LogitRegression(learning_rate = 0.01, epochs = 999)  # Si Funciona pero necesita muchas MUCHAS Interaciones como tambien el
model.fit(X_train, Y_train)  # tarda su rato dependiendo las epocas y el ratio de apredizaje que se le ponga

# Prediction on test set
Y_pred = model.predict(X_test)

print(f'Prediccion Modelo realizado de la tercera prueba:\n {Y_pred}')  # Checamos por los primeros 5 valores que usaramos como predictores
print("/------------------------------------/")

# Reporte de Clasificacion  sin cross validation del modelo generado
print(f'Reporte de clasificacion sin crossvalidation de la tercera prueba :\n {classification_report(Y_test, Y_pred)}')
print("/------------------------------------/")

#------------------------------------------------------------------------------#
#                                CUARTA PRUEBA                                #
#------------------------------------------------------------------------------#

# MODEL DEPLOYMENT

# Preidicion de quien salio del puerto  S (S = Southampton)
# Rompemos el dataframe para  el set entrenamiento (4/5 de datos del dataframe) y  set pruebas (pass set) (1/5 de datos del dataframe)
# y quitamos la variable Survived qque es la que queremos checar
X_train, X_test, Y_train, Y_test = train_test_split(titanic_data_dmy.drop(['S'], axis= 1), # Valores en X
                                                    titanic_data_dmy['S'], test_size = 0.2, # Valores en Y
                                                    random_state= 200) # seed de random para tener los mismos resultados


#------------------------------------------------------------------------------#
#                           SECCION SIN USO LIBRERIA                           #
#------------------------------------------------------------------------------#
# Llamando al modelo
# Model training
model = LogitRegression(learning_rate = 0.01, epochs = 999)  # Si Funciona pero necesita muchas MUCHAS Interaciones como tambien el
model.fit(X_train, Y_train)  # tarda su rato dependiendo las epocas y el ratio de apredizaje que se le ponga

# Prediction on test set
Y_pred = model.predict(X_test)

print(f'Prediccion Modelo realizado de la tercera prueba:\n {Y_pred}')  # Checamos por los primeros 5 valores que usaramos como predictores
print("/------------------------------------/")

# Reporte de Clasificacion  sin cross validation del modelo generado
print(f'Reporte de clasificacion sin crossvalidation de la tercera prueba :\n {classification_report(Y_test, Y_pred)}')
print("/------------------------------------/")