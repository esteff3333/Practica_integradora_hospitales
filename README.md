# Proyecto integrador hospitales

### 1.Planteamiento de la problemática
Hemos sido contratados en el equipo de ciencia de datos en una consultora de renombre. Nos han asignado a un proyecto de estudio de atención en salud para un
importante hospital. Nuestro cliente desea saber las características más importantes que tienen los pacientes de cierto tipo de enfermedad que terminan en
hospitalización. Fue definido como caso aquel paciente que fue sometido a biopsia prostática y que en un periodo máximo de 30 días posteriores al procedimiento
presentó fiebre, infección urinaria o sepsis; requiriendo manejo médico ambulatorio u hospitalizado para la resolución de la complicación y como control al paciente
que fue sometido a biopsia prostática y que no presentó complicaciones infecciosas en el período de 30 días posteriores al procedimiento. Dado que tienen en su
base de datos algunos datos referentes a los pacientes y resultados de exámenes diagnósticos, de pacientes hospitalizados y no hospitalizados, nos han entregado
esta información.
Para ello, nuestro departamento de datos ha recopilado Antecedentes del paciente , Morbilidad asociada al paciente y Antecedentes relacionados
con la toma de la biopsia y Complicaciones infecciosas .
### 2. Preparación de datos
```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
```

Cargamos la base de datos
```python
df = pd.read_csv('/content/BBDD_Hospitalización.csv', sep = ";")
df
df.info()
df.dtypes 
df.describe()
```

¿Cuanto es el porcentaje de valores nulos?

```python
(df.isnull().melt().pipe(lambda df:(sns.displot(data=df,y='variable',hue='value',multiple='fill',aspect=4))))
plt.show()
```

Este codigo nos devuelve los indices que tienen valores nulos en cualquier columna
```python
indices_nulos = df[df.isnull().any(axis=1)].index
indices_nulos
Cantidad de filas que tienen valores nulos
len(indices_nulos)
```

Miramos cuantos valores nulos tiene cada fila
```python
df_filtrado = df.iloc[indices_nulos, :]
valores_nulos_por_fila = df_filtrado.isnull().sum(axis=1)[indices_nulos]
valores_nulos_por_fila
```

Con esto, decidimos que las filas 568 y 569 pueden ser eliminadas ya que no tienen ningún 
valor 
```python
df2 = df.drop(index= [568, 569]) 
```
Revisamos los nulos en la columna Hospitalizacion ya que es nuestra variable objetivo. Y luego eliminamos las filas que tienen nulos 

```python
df2['HOSPITALIZACION'].isnull().sum()
df2=df2.dropna(subset='HOSPITALIZACION')
df2.info()
```

Modificamos los tipos de datos de las columnas que consideramos que no podían ser floats. Por ejemplo la edad, el número de muestras y los días.

```python
df2['EDAD'] = df2['EDAD'].astype(int)
df2['NUMERO DE MUESTRAS TOMADAS'] = df2['NUMERO DE MUESTRAS TOMADAS'].astype(int)
df2['DIAS HOSPITALIZACION MQ'] = df2['DIAS HOSPITALIZACION MQ'].astype(int)
df2['DIAS HOSPITALIZACIÓN UPC'] = df2['DIAS HOSPITALIZACIÓN UPC'].astype(int)
```

Modificamos los nombres de dos columnas poco prácticas para su uso.

```python
df2.rename(columns={'NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACIÓN INFECCIOSA': 'NRO DIAS POST BIOPSIA COMPLICACION INFECCIOSA'}, inplace=True)
df2.rename(columns={'ANTIBIOTICO UTILIAZADO EN LA PROFILAXIS': 'ANTIBIOTICO'}, inplace=True)
```

Hacemos un boxplot e histogramas de las variables para descubrir valores atipicos y etender si son errores al cargar la base de datos

```python
sns.boxplot(data=df2['DIAS HOSPITALIZACION MQ'])
df2["DIAS HOSPITALIZACION MQ"].hist()  
plt.title('DIAS HOSPITALIZACION MQ')
plt.xlabel('DIAS HOSPITALIZACION MQ')
plt.ylabel('Frecuencia')
```

En este gráfico podemos observar que la mayoría de los registros son "0", por lo tanto la mayoría de los pacientes no fueron hospitalizados para tratamiento médico quirúrgico. Los valores atípicos en este caso no representan valores equivocados. 

```python
sns.boxplot(data=df2['DIAS HOSPITALIZACIÓN UPC'])
df2["DIAS HOSPITALIZACIÓN UPC"].hist()  
plt.title('DIAS HOSPITALIZACIÓN UPC')
plt.xlabel('DIAS HOSPITALIZACIÓN UPC')
plt.ylabel('Frecuencia')
```

En este gráfico podemos observar que la mayoría de los registros son "0", por lo tanto la mayoría de los pacientes no fueron hospitalizados para tratamiento de estado crítico. Los valores atípicos en este caso no representan valores equivocados. 

```python
sns.boxplot(data=df2['NUMERO DE MUESTRAS TOMADAS'])
df2["NUMERO DE MUESTRAS TOMADAS"].hist()  
plt.title('NUMERO DE MUESTRAS TOMADAS')
plt.xlabel('NUMERO DE MUESTRAS TOMADAS')
plt.ylabel('Frecuencia')
```

En este gráfico podemos observar valores atípicos, pero son pertinentes a la cantidad de muestras tomadas.

``` python
df2["PSA"].hist()
plt.title('PSA')
plt.xlabel('PSA')
plt.ylabel('Frecuencia')
```

En este gráfico aparentemente observamos un gran número de valores atípicos, pero investigamos la concentración del PSA en la sangre y los valores altos pueden corresponder a pacientes con cáncer de próstata o con metástasis. 
Fuente: https://urologiapepauguet.com/blog/psa-valor-normal-causas-elevacion-tipos/#:~:text=El%20%C3%BAnico%20caso%20en%20el,o%202000%20ng%2Fml).

```python
sns.boxplot(data=df2['EDAD'])
```

En este gráfico podemos observar 2 valores atípicos por encima de 140. Entendemos que es un error humano y que debemos tomar una decisión sobre qué hacer. 
Hablamos con el cliente, le entregamos el informe de los outliers y decidimos normalizar los dos valores con el promedio de la columna "EDAD". Buscamos los dos registros y los modificamos. 

```python
df2 [df2["EDAD"] >100]
```

Con este código creamos una máscara en la cual sacamos los valores mayores a 100, luego creamos la media y reemplazamos los valores con el valor de la media(64). 

```python
round(df2[df2["EDAD"] < 100]["EDAD"].mean())
df2.loc[[161, 181], "EDAD"] = round(df2[df2["EDAD"] < 100]["EDAD"].mean())
```

Revisar valores faltantes o nulos 

```python
num_vars = df2.select_dtypes(include=['float', 'int']).columns
null_counts = df2[num_vars].isnull().sum()

# Imprime los resultados
print("Cantidad de valores nulos o NaN en variables numéricas:")
print(null_counts)
```

Rellenamos los valores nulos de PSA con la media 

```python
media = df2['PSA'].mean()
df2['PSA'] = df2['PSA'].fillna(media)
```

Miramos los valores únicos de cada columna categórica para normalizarlas
```python
valores_unicos = df2.apply(lambda x: x.unique())
valores_unicos

Con la función str.strip() quitamos los espacios antes y después de los nombres de los valores.

```
Columnas = ["ENF. CRONICA PULMONAR OBSTRUCTIVA", "BIOPSIA", "TIPO DE CULTIVO", "AGENTE AISLADO", "PATRON DE RESISTENCIA"]
for i in Columnas:
  df2[i] = df[i].str.strip()
```

Miramos columna por columna que nos interesan los valores unicos para ver que sean iguales o si necesitan normalizarse

```
Columnas = ["ANTIBIOTICO", "ENF. CRONICA PULMONAR OBSTRUCTIVA", "BIOPSIA", "TIPO DE CULTIVO", "AGENTE AISLADO", "PATRON DE RESISTENCIA"]
```
```
for columna in Columnas:
    valores_unicos = df2[columna].unique()
    print(f"Valores únicos de la columna '{columna}':")
    for valor in valores_unicos:
        print(valor)
    print()
    ```

Normalizamos los valores de la columna antibiotico y patrón de resistencia
df2["ANTIBIOTICO"] = df2["ANTIBIOTICO"].

```python
replace("FLUOROQUINOLONA_AMINOGLICÓSIDO","FLUOROQUINOLONA_AMINOGLICOSIDO")
df2["PATRON DE RESISTENCIA"] = df2["PATRON DE RESISTENCIA"].replace("RESISTENTE A AMPI, CIPRO Y GENTA","AMPI, CIPRO Y GENTA")
df2["PATRON DE RESISTENCIA"] = df2["PATRON DE RESISTENCIA"].replace("RESISTENTE A AMPI, SULFA, CEFADROXILO, CEFUROXIMO, CIPRO Y CEFEPIME, CEFOTAXIMA","AMPI, SULFA, CEFADROXILO, CEFUROXIMO, CIPRO Y CEFEPIME, CEFOTAXIMA")
```

Vamos a eliminar las columnas DIAS HOSPITALIZACIÓN UPC y DIAS HOSPITALIZACIÓN MQ ya que son redundantes con la columna Hospitalización, pero primero corroboramos que en esta última columna tengan valor SI, ya que son personas que fueron hospitalizadas

```python
df2[(df2["DIAS HOSPITALIZACION MQ"] != 0) | (df2["DIAS HOSPITALIZACIÓN UPC"] != 0)]
df2 = df2.drop(["DIAS HOSPITALIZACION MQ", "DIAS HOSPITALIZACIÓN UPC"], axis = 1)
```

Realizamos gráficos de las columnas categóricas

```python
Columnas = ["ANTIBIOTICO", "BIOPSIA", "TIPO DE CULTIVO", "AGENTE AISLADO", "PATRON DE RESISTENCIA"]
```

Crear el gráfico de barras

```python
plt.barh(df2["ANTIBIOTICO"].value_counts().index, df2["ANTIBIOTICO"].value_counts().values)
plt.xlabel('Cantidad')
plt.ylabel('Tipo de antibiotico')
plt.title('Conteo de tipos de antibioticos')
plt.show()
```

En los siguientes gráficos dejamos fuera la columna que hace refencia a los pacientes que no se le han realizado los estudios ya que siendo tantos nos dificultaba ver los datos de las otras columnas

```python
df3 = df2[df2["BIOPSIA"] != "NEG"]
# Crear el gráfico de barras
plt.barh(df3["BIOPSIA"].value_counts().index, df3["BIOPSIA"].value_counts().values)
plt.xlabel('Cantidad')
plt.ylabel('Resultado de biopsia')
plt.title('Conteo de resultados de biopsia')
plt.show()

df3 = df2[df2["TIPO DE CULTIVO"] != "NO"]
plt.barh(df3["TIPO DE CULTIVO"].value_counts().index, df3["TIPO DE CULTIVO"].value_counts().values)

plt.xlabel('Cantidad')
plt.ylabel('Tipo de cultivo pedido')
plt.title('Conteo de tipos de cultivos')

plt.show()
df3 = df2[df2["AGENTE AISLADO"] != "NO"]
plt.barh(df3["AGENTE AISLADO"].value_counts().index, df3["AGENTE AISLADO"].value_counts().values)

plt.xlabel('Cantidad')
plt.ylabel('Tipo de agente aislado')
plt.title('Conteo de tipos de cultivos')

plt.show()
df3 = df2[df2["PATRON DE RESISTENCIA"] != "NO"]
plt.barh(df3["PATRON DE RESISTENCIA"].value_counts().index, df3["PATRON DE RESISTENCIA"].value_counts().values)

plt.xlabel('Cantidad')
plt.ylabel('Tipo de agente aislado')
plt.title('Conteo de tipos de cultivos')
plt.show()


Ahora vamos atransformar los datos de las columnas de object a numericos

```
for columna in df2.select_dtypes(include='object'):
 valores_unicos = df2[columna].unique()
 print(f"Valores únicos en la columna '{columna}': {valores_unicos}")
 ```

Cambiamos los valores de Si y No por 1 y 0 según corresponda

```
diccionario = {"NO": 0, "SI": 1}
df2["DIABETES2"] = df2.DIABETES.map(diccionario)
df2["HOSPITALIZACIÓN ULTIMO MES2"] = df2["HOSPITALIZACIÓN ULTIMO MES"].map(diccionario)
df2["BIOPSIAS PREVIAS2"] = df2["BIOPSIAS PREVIAS"].map(diccionario)
df2["VOLUMEN PROSTATICO2"] = df2["VOLUMEN PROSTATICO"].map(diccionario)
df2["CUP2"] = df2["CUP"].map(diccionario)
df2["FIEBRE2"] = df2["FIEBRE"].map(diccionario)
df2["ITU2"] = df2["ITU"].map(diccionario)
df2["HOSPITALIZACION2"] = df2["HOSPITALIZACION"].map(diccionario)
df2['NRO DIAS POST BIOPSIA COMPLICACION INFECCIOSA'] = df2['NRO DIAS POST BIOPSIA COMPLICACION INFECCIOSA'].replace('NO',0)
```

Cambiamos las variables categóricas a numéricas 

```
df2 = pd.concat([df2, pd.get_dummies(df2["ANTIBIOTICO"])], axis =1)
df2 = pd.concat([df2, pd.get_dummies(df2["ENF. CRONICA PULMONAR OBSTRUCTIVA"])], axis =1)
df2 = pd.concat([df2, pd.get_dummies(df2["BIOPSIA"])], axis =1)
df2 = pd.concat([df2, pd.get_dummies(df2["TIPO DE CULTIVO"])], axis =1)
df2 = pd.concat([df2, pd.get_dummies(df2["AGENTE AISLADO"])], axis =1)
df2 = pd.concat([df2, pd.get_dummies(df2["PATRON DE RESISTENCIA"])], axis =1)
df2.columns
df2.head()
# = pd.get_dummies(df2['FIEBRE'])

```
Debemos aplicar el código anterior a las columnas que nos interesan. 
# Calcular la matriz de correlación para las variables seleccionadas

```
correlation_matrix = df2[['DIABETES2', 'HOSPITALIZACIÓN ULTIMO MES2',
       'BIOPSIAS PREVIAS2', 'VOLUMEN PROSTATICO2', 'CUP2', 'FIEBRE2', 'ITU2',
       'HOSPITALIZACION2']].corr()
```

Crear el gráfico de correlación
```
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```

# Mostrar el gráfico de correlacion
```
plt.title('Gráfico de Correlación')
plt.show()

Calcular la matriz de correlación para las variables seleccionadas
correlation_matrix = df2[['HOSPITALIZACION2', 'CEFALOSPORINA_AMINOGLUCOCIDO',
       'FLUOROQUINOLONA_AMINOGLICOSIDO', 'OROQUINOLONAS', 'OTROS',  'SI',
       'SI, ASMA', 'SI, EPOC']].corr()
```

Crear el gráfico de correlación
```
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```

Mostrar el gráfico
```
plt.title('Gráfico de Correlación entre hospitalizacion y antibioticos')
plt.show()
```

Calcular la matriz de correlación para las variables seleccionadas

```
correlation_matrix = df2[['ADENOCARCINOMA GLEASON 10',
       'ADENOCARCINOMA GLEASON 6', 'ADENOCARCINOMA GLEASON 7',
       'ADENOCARCINOMA GLEASON 8', 'ADENOCARCINOMA GLEASON 9',
       'CARCINOMA INDIFERENCIADO DE CELULAS CLARAS', "HOSPITALIZACION2"]].corr()
```

Crear el gráfico de correlación
```
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```

Mostrar el gráfico
```
plt.title('Gráfico de Correlación entre hospitalizacion y antibioticos')
plt.show()
df2.columns
plt.figure(figsize=(20, 12))
sns.heatmap(
 data=df2.corr(),
 cmap=sns.diverging_palette(20,230,as_cmap=True),
 center=0,
 vmin=-1,
 vmax=1,
 linewidths=0.5,
 annot=True
)
plt.show()
plt.scatter(df2['PSA'], df2['BIOPSIA'])
plt.scatter(df2['DIAS HOSPITALIZACION MQ'], df2['PSA'])
sns.barplot(y='BIOPSIA', x='EDAD', data=df)
plt.scatter(df2['DIAS HOSPITALIZACION MQ'], df2['FIEBRE'])
plt.scatter(df2['DIAS HOSPITALIZACION MQ'], df2['EDAD'])
```

Contar los valores de 'sí' y 'no'
```
counts = df2['HOSPITALIZACION'].value_counts()
```

Crear el gráfico de barras
```
plt.bar(counts.index, counts.values)
```

Agregar etiquetas al gráfico
```
plt.xlabel('Hospitalización')
plt.ylabel('Cantidad')
plt.title('Conteo de hospitalizaciones')
```

Mostrar el gráfico
```
plt.show()
counts = df2['DIAS HOSPITALIZACION MQ'].value_counts()

plt.bar(counts.index, counts.values)
plt.xlabel('Hospitalización')
plt.ylabel('Cantidad')
plt.title('Conteo de hospitalizaciones')
plt.show()

counts = df2['DIAS HOSPITALIZACIÓN UPC'].value_counts()

plt.bar(counts.index, counts.values)
plt.xlabel('Hospitalización')
plt.ylabel('Cantidad')
plt.title('Conteo de hospitalizaciones')
plt.show()

counts = df2['HOSPITALIZACIÓN ULTIMO MES'].value_counts()

plt.bar(counts.index, counts.values)
plt.xlabel('Hospitalización')
plt.ylabel('Cantidad')
plt.title('Conteo de hospitalizaciones')
plt.show()

counts = df2['DIABETES'].value_counts()

plt.bar(counts.index, counts.values)
plt.xlabel('Hospitalización')
plt.ylabel('Cantidad')
plt.title('Conteo de hospitalizaciones')
plt.show()

counts = df2['BIOPSIAS PREVIAS'].value_counts()

plt.bar(counts.index, counts.values)
plt.xlabel('Hospitalización')
plt.ylabel('Cantidad')
plt.title('Conteo de hospitalizaciones')
plt.show()

counts = df2['VOLUMEN PROSTATICO'].value_counts()

plt.bar(counts.index, counts.values)
plt.xlabel('Hospitalización')
plt.ylabel('Cantidad')
plt.title('Conteo de hospitalizaciones')
plt.show()
``` 
Convertimos la columna fiebre a binaria
