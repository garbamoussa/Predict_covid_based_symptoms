# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC # Modélisation 
# MAGIC 
# MAGIC Le projet choisit est un sujet de type de la santé en rapport avec la crise sanitaire COVID 19 . Il s'agit d'un apprentissage de type suppervisé d'une problématique de classification de patients présentant de symptomes simmilaires au COVID 19. Le but est d'indentifer les patients atteints du COVID 19. 
# MAGIC ## Sommaire
# MAGIC 
# MAGIC ### I.	Gathering data 
# MAGIC 
# MAGIC 1. Open data Health 
# MAGIC 
# MAGIC 2. Problématique et identification de données 
# MAGIC 
# MAGIC 
# MAGIC ### II. ANALYSE EXPLORATOIRE DES DONNEES
# MAGIC 1.	Echantillonnage
# MAGIC 
# MAGIC 2.	Features engineering 
# MAGIC 
# MAGIC 3.	Visualisation des données
# MAGIC 
# MAGIC ### III.	PREPARATION DES DONNEES 
# MAGIC 
# MAGIC 1.	Répartition de données en features X et Target Y
# MAGIC 
# MAGIC     1.1 Features 
# MAGIC     
# MAGIC     1.2 Target 
# MAGIC     
# MAGIC     
# MAGIC 2.	Selection des variables
# MAGIC 
# MAGIC ### IV.	MODELISATION 
# MAGIC 
# MAGIC 1. Description du process de modélisation
# MAGIC 2.	Hyperparameters tuning
# MAGIC 
# MAGIC 3.	Evaluation 
# MAGIC 
# MAGIC ### V.	CONCLUSION

# COMMAND ----------

# MAGIC %sh
# MAGIC #pip install matplotlib
# MAGIC #pip install numpy
# MAGIC #pip install pandas==0.25.1
# MAGIC ###conda install -c anaconda basemap
# MAGIC pip install scikit-learn
# MAGIC pip install spark_df_profiling
# MAGIC pip install pydotplus
# MAGIC 
# MAGIC pip install missingno
# MAGIC pip install graphviz
# MAGIC pip install xgboost 
# MAGIC pip install imblearn
# MAGIC 
# MAGIC pip install --upgrade scikit-learn

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Importation de librairies 

# COMMAND ----------

# Import libraries

from sklearn import tree
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,BayesianRidge,Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import pydotplus
from re import sub
import scipy.stats as sci
import seaborn as sns
import spark_df_profiling
from pathlib import Path


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import SVG
from graphviz import Source
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
 
# Set visualization prefrences 
sns.set(font_scale=1.5, style="darkgrid")
pd.set_option('display.max_columns', None)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## 1. Gathering data 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Importation de données 
# MAGIC 
# MAGIC 
# MAGIC %md 
# MAGIC 
# MAGIC ### Open data Health 
# MAGIC 
# MAGIC - Competition :  Diagnosis of COVID -19 and its clinician spectrum 
# MAGIC 
# MAGIC https://www.kaggle.com/einsteindata4u/covid19

# COMMAND ----------



## Importer les données 
# File location and type
file_location_covid_einstein = "dbfs:/FileStore/tables/covid_einstein.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

## import all_vars_for_zeroinf_analysis  file 
# The applied options are for CSV files. For other file types, these will be ignored.
Covid_einstein = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location_covid_einstein)



# COMMAND ----------

# MAGIC 
# MAGIC %md 
# MAGIC 
# MAGIC ### Problématique et identification de données
# MAGIC 
# MAGIC -  L'identifier les données de santé permettant de caractériser les patients présentant préliminaires. 
# MAGIC - 	L’implémentation de ces algorithmes de machine Learning capables d’apprendre à partir de données et de déterminer la probabilité du personne soit atteinte ou pas du COVID 19 à partir de ses symptômes et d'autres caractéristiques.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### II. ANALYSE EXPLORATOIRE DES DONNEES

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Echantillonnage 

# COMMAND ----------


pandas_Covid_einstein = Covid_einstein.toPandas()

# COMMAND ----------

pandas_Covid_einstein.head()

# COMMAND ----------

### Taille et nomre de colonnes

pandas_Covid_einstein.shape

# COMMAND ----------

pandas_Covid_einstein.isnull().sum()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 
# MAGIC ce dataset  contient des données anonymisées de patients de l'Hôpital Israelita Albert Einstein, à São Paulo, au Brésil, et qui avaient des échantillons collectés pour effectuer la RT-PCR SARS-CoV-2 et des tests de laboratoire supplémentaires lors d'une visite médicale.
# MAGIC 
# MAGIC Toutes les données ont été anonymisées conformément aux meilleures pratiques et recommandations internationales. Toutes les données cliniques ont été normalisées pour avoir une moyenne de zéro et un écart-type unitaire.
# MAGIC Nous avons les variables du dataset contenant les symptômes des patients, les caractérisques socio, le pays ... 
# MAGIC 
# MAGIC Parmi les variables : 
# MAGIC 
# MAGIC - Respiratory Syncytial Virus : 
# MAGIC 
# MAGIC - CoronavirusNL63 :  
# MAGIC 
# MAGIC - Patient age quantile 
# MAGIC 
# MAGIC - Patient addmited to intensive care unit (1=yes, 0=no)  
# MAGIC 
# MAGIC -  Monocytes   
# MAGIC 
# MAGIC - Relationship (Patient/Normal)  
# MAGIC 
# MAGIC - Leukocytes

# COMMAND ----------

#Define missing data function to identify the total number of missing data and associated percentage 
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))

# COMMAND ----------

pandas_Covid_einstein.head()

# COMMAND ----------

pandas_Covid_einstein.columns

# COMMAND ----------

pandas_Covid_einstein.shape

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Nous avons 5644 lignes et 111 variables dans le dataset 

# COMMAND ----------

# Check difference between cases and confirmed cases
print('NAs in ARS-Cov-2 exam result: ', pandas_Covid_einstein['SARS-Cov-2 exam result'].isna().sum())
print('NAs in Patient addmited to regular ward (1=yes, 0=no): ', pandas_Covid_einstein['Patient addmited to regular ward (1=yes, 0=no)'].isna().sum())

print('% different ARS-Cov-2 exam result: ', sum(pandas_Covid_einstein['SARS-Cov-2 exam result']!=pandas_Covid_einstein['Patient addmited to regular ward (1=yes, 0=no)'])/pandas_Covid_einstein.shape[0]*100)




# COMMAND ----------

missing_data(pandas_Covid_einstein)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Identifier les variables présentant un pourcentage élévé de données manquantes 
# MAGIC 
# MAGIC Par exemple, pour la variable : 
# MAGIC 
# MAGIC - Hematocrit 5041 lignes manquantes et un pourcentage 89%. 
# MAGIC 
# MAGIC - Hemoglobin 5042 lignes manquantes et un pourcentage 89%. 
# MAGIC 
# MAGIC - Lymphocytes 5042 lignes manquantes et un pourcentage 89%. 
# MAGIC 
# MAGIC - Mean corpuscular hemoglobin (MCH) 5042 lignes manquantes et un pourcentage 89%. 
# MAGIC 
# MAGIC - CoronavirusNL63 4292 lignes manquantes et un pourcentage 76%.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Analyse de la variable target 
# MAGIC 
# MAGIC Variable Target SARS-Cov-2 exam result : La variable prend "Positive"  et "negative" pour le test. 

# COMMAND ----------

pandas_Covid_einstein['SARS-Cov-2 exam result'].value_counts().plot.barh()

# COMMAND ----------

## Regrouper les variables par rapport à la target 

percent_target = pandas_Covid_einstein.groupby('SARS-Cov-2 exam result').count()
percent_target['percent'] = 100*(percent_target['Leukocytes']/pandas_Covid_einstein['SARS-Cov-2 exam result'].count())
percent_target.reset_index(level=0, inplace=True)
percent_target

# COMMAND ----------

import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'negative','positive'
sizes = [9, 1]
explode = (0.1, 0)  # only "explode" the 1st slice (i.e. 'Toxic contents')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.0f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC Nous avons 90% de test negatives et 10 % de positives dans notre échantillon 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Features engineering 

# COMMAND ----------

# les valeurs manquantes 
msno.bar(pandas_Covid_einstein, figsize=(16, 4))

# COMMAND ----------

(pandas_Covid_einstein.isnull().sum() == pandas_Covid_einstein.shape[0]).any()

# COMMAND ----------

full_null_data = (pandas_Covid_einstein.isnull().sum() == pandas_Covid_einstein.shape[0])
full_null_columns = full_null_data[full_null_data == True].index

# COMMAND ----------

## colonnes avec toutes les valeurs égales à 0  

print(full_null_columns.tolist())

# COMMAND ----------

pandas_Covid_einstein.drop(full_null_columns, axis=1, inplace=True)

# COMMAND ----------

(pandas_Covid_einstein.isnull().sum() / pandas_Covid_einstein.shape[0]).sort_values(ascending=False).head()

# COMMAND ----------

contain_null_series = (pandas_Covid_einstein.isnull().sum() != 0).index

# COMMAND ----------

target = 'SARS-Cov-2 exam result'
just_one_target = []

for col in contain_null_series:
    i = pandas_Covid_einstein[pandas_Covid_einstein[col].notnull()][target].nunique()
    if i == 1:
        just_one_target.append(col)    

# Selection de colonne contenant uniquement covid négative        
print(just_one_target)

# COMMAND ----------

for col in just_one_target:
    print(pandas_Covid_einstein[pandas_Covid_einstein[col].notnull()][target].unique())

# COMMAND ----------

pandas_Covid_einstein.drop(just_one_target, axis=1, inplace=True)

# COMMAND ----------

msno.bar(pandas_Covid_einstein, figsize=(16, 4))

# COMMAND ----------

not_null_series = (pandas_Covid_einstein.isnull().sum() == 0)
not_null_columns = not_null_series[not_null_series == True].index
not_null_columns = not_null_columns[1:]

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Visualisation des données

# COMMAND ----------

def plot_histograms(pandas_Covid_einstein, cols, subplots_rows, subplots_cols, figsize=(20, 10), target='SARS-Cov-2 exam result'):
    df_neg = pandas_Covid_einstein[pandas_Covid_einstein[target] == 'negative']
    df_pos = pandas_Covid_einstein[pandas_Covid_einstein[target] == 'positive']
    
    cols = cols.tolist()
    cols.remove(target)
    
    plt.figure()
    fig, ax = plt.subplots(subplots_rows, subplots_cols, figsize=figsize)
    
    i = 0    
    for col in cols:
        i += 1
        plt.subplot(subplots_rows, subplots_cols, i)
        sns.distplot(df_neg[col], label="Negative", bins=15, kde=False)
        sns.distplot(df_pos[col], label="Positive", bins=15, kde=False)
        plt.legend()
    plt.show()
    
plot_histograms(pandas_Covid_einstein, not_null_columns, 2, 2)

# COMMAND ----------

# variables  categorical
mask_pos_neg = {'positive': 1, 'negative': 0}
mask_detected = {'detected': 1, 'not_detected': 0}
mask_notdone_absent_present = {'not_done': 0, 'absent': 1, 'present': 2}
mask_normal = {'normal': 1}
mask_urine_color = {'light_yellow': 1, 'yellow': 2, 'citrus_yellow': 3, 'orange': 4}
mask_urine_aspect = {'clear': 1, 'lightly_cloudy': 2, 'cloudy': 3, 'altered_coloring': 4}
mask_realizado = {'Não Realizado': 0}
mask_urine_leuk = {'<1000': 1000}
mask_urine_crys = {'Ausentes': 1, 'Urato Amorfo --+': 0, 'Oxalato de Cálcio +++': 0,
                   'Oxalato de Cálcio -++': 0, 'Urato Amorfo +++': 0}

# COMMAND ----------

pandas_Covid_einstein = pandas_Covid_einstein.replace(mask_detected)
pandas_Covid_einstein = pandas_Covid_einstein.replace(mask_pos_neg)
pandas_Covid_einstein = pandas_Covid_einstein.replace(mask_notdone_absent_present)
pandas_Covid_einstein = pandas_Covid_einstein.replace(mask_normal)
pandas_Covid_einstein = pandas_Covid_einstein.replace(mask_realizado)
pandas_Covid_einstein = pandas_Covid_einstein.replace(mask_urine_leuk)
pandas_Covid_einstein = pandas_Covid_einstein.replace(mask_urine_color)
pandas_Covid_einstein = pandas_Covid_einstein.replace(mask_urine_aspect)
pandas_Covid_einstein = pandas_Covid_einstein.replace(mask_urine_crys)

pandas_Covid_einstein['Urine - pH'] = pandas_Covid_einstein['Urine - pH'].astype('float')
pandas_Covid_einstein['Urine - Leukocytes'] = pandas_Covid_einstein['Urine - Leukocytes'].astype('float')

# COMMAND ----------

corr=pandas_Covid_einstein.corr(method='pearson')
corr

# COMMAND ----------


corr=pandas_Covid_einstein.corr(method='pearson')
corr=corr.sort_values(by=["SARS-Cov-2 exam result"],ascending=False).iloc[0].sort_values(ascending=False)
plt.figure(figsize=(15,20))
sns.barplot(x=corr.values, y=corr.index.values);
plt.title("Correlation Plot at State Level")
display()

# COMMAND ----------

display(pandas_Covid_einstein)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### III.	PREPARATION DES DONNEES 

# COMMAND ----------

x = pandas_Covid_einstein.drop(['Patient ID', 'SARS-Cov-2 exam result'], axis=1)
x.fillna(999999, inplace=True)
y = pandas_Covid_einstein['SARS-Cov-2 exam result']

# COMMAND ----------

dt = DecisionTreeClassifier(max_depth=3)

# COMMAND ----------

dt.fit(x, y)

# COMMAND ----------

dt_feat = pd.DataFrame(dt.feature_importances_, index=x.columns, columns=['feat_importance'])
dt_feat.sort_values('feat_importance').tail(8).plot.barh()
plt.show()


# COMMAND ----------

# MAGIC %md 
# MAGIC Feature importance est très important dans le domaine de  la modélisation machine learning. Il permet de comprendre quelles sont les variables qui contribuent le plus dans le modèle et aussi de pouvoir interpreter les résultats. 
# MAGIC 
# MAGIC Cette partie est cruciale car elle va permettre aux data scientist de pouvoir expliquer aisément les résultats. 
# MAGIC Les variables apparaissant dans les features importance : 
# MAGIC  
# MAGIC 
# MAGIC - Leukocytes 
# MAGIC 
# MAGIC - Patient addmited to regular ward (1 =yes, 0=no)
# MAGIC 
# MAGIC - Patient age quantile 
# MAGIC 
# MAGIC - Basophils

# COMMAND ----------

sns.distplot(pandas_Covid_einstein[pandas_Covid_einstein['SARS-Cov-2 exam result'] == 1]['Leukocytes'], label="Covid")
sns.distplot(pandas_Covid_einstein[pandas_Covid_einstein['SARS-Cov-2 exam result'] == 0]['Leukocytes'], label="No Covid")
plt.legend()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ###  IV.	MODELISATION 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Description du process de modélisation
# MAGIC 
# MAGIC 
# MAGIC Nou testerons dans cette section plusieurs modèles de classification tels les SVM, les methodes ensembles (RF, adaboost,...), Reseaux de neurones etc... dans l'objectif de choisir le meilleur modèle. Puis nous optimiserons les hyperparametres des modèles qui nous semblent les plus performants. Enfin nous generons les courbes d'apprentissage afn d'évaluer le niveau d'apprentissage de ces modèles pour verifer l'overfitting où  l'underfitting. 

# COMMAND ----------

classifiers = {'Logistic Regression' : LogisticRegression(),
               'KNN': KNeighborsClassifier(),
               'Decision Tree': DecisionTreeClassifier(),
               'Random Forest': RandomForestClassifier(),
               'AdaBoost': AdaBoostClassifier(),
               'SVM': SVC()}

samplers = {'Random_under_sampler': RandomUnderSampler(),
            'Random_over_sampler': RandomOverSampler()}

drop_cols = ['Patient ID', 'Patient addmited to regular ward (1=yes, 0=no)',
             'Patient addmited to semi-intensive unit (1=yes, 0=no)',
             'Patient addmited to intensive care unit (1=yes, 0=no)']

# COMMAND ----------

def df_split(pandas_Covid_einstein, target='SARS-Cov-2 exam result', drop_cols=drop_cols):
    pandas_Covid_einstein = pandas_Covid_einstein.drop(drop_cols, axis=1)
    pandas_Covid_einstein = pandas_Covid_einstein.fillna(999)
    x = pandas_Covid_einstein.drop(target, axis=1)
    y = pandas_Covid_einstein[target]    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)                          
    return x_train, x_test, y_train, y_test

def train_clfs(pandas_Covid_einstein, classifiers, samplers):
    
    x_train, x_test, y_train, y_test = df_split(pandas_Covid_einstein)
    
    names_samplers = []
    names_clfs = []
    results_train_cv_roc_auc = []
    results_train_cv_recall = []
    results_train_cv_accuracy = []
    results_test_roc_auc = []
    results_test_recall = []
    results_test_accuracy = []
    
    for name_sampler, sampler in samplers.items():
        print(f'Sampler: {name_sampler}\n')
        for name_clf, clf in classifiers.items():
            print(f'Classifier: {name_clf}\n')
            
            pipeline = Pipeline([('sampler', sampler),
                                 ('clf', clf)])
            
            cv_auc = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='roc_auc') 
            cv_rec = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='recall')                                
            cv_acc = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='accuracy')        

            pipeline.fit(x_train, y_train)        
            y_pred = pipeline.predict(x_test)
            
            names_samplers.append(name_sampler)
            names_clfs.append(name_clf)
            results_train_cv_roc_auc.append(cv_auc)
            results_train_cv_recall.append(cv_rec)
            results_train_cv_accuracy.append(cv_acc)
            results_test_roc_auc.append(roc_auc_score(y_test, y_pred))
            results_test_recall.append(recall_score(y_test, y_pred))
            results_test_accuracy.append(accuracy_score(y_test, y_pred))

            print(f'CV\t-\troc_auc:\t{round(cv_auc.mean(), 3)}')
            print(f'CV\t-\trecall:\t\t{round(cv_rec.mean(), 3)}')
            print(f'CV\t-\taccuracy:\t{round(cv_acc.mean(), 3)}')

            print(f'Test\t-\troc_auc:\t{round(roc_auc_score(y_test, y_pred), 3)}')         
            print(f'Test\t-\trecall:\t\t{round(recall_score(y_test, y_pred), 3)}')          
            print(f'Test\t-\taccuracy:\t{round(accuracy_score(y_test, y_pred), 3)}')      
            print('\n<-------------------------->\n')

    df_results_test = pd.DataFrame(index=[names_clfs, names_samplers], columns=['ROC_AUC', 'RECALL', 'ACCURACY'])
    df_results_test['ROC_AUC'] = results_test_roc_auc
    df_results_test['RECALL'] = results_test_recall
    df_results_test['ACCURACY'] = results_test_accuracy

    return df_results_test

# COMMAND ----------

df_results_test = train_clfs(pandas_Covid_einstein, classifiers, samplers)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Hyperparameters tuning

# COMMAND ----------

def train_xgb(pandas_Covid_einstein, clf):
    
    x_train, x_test, y_train, y_test = df_split(pandas_Covid_einstein)

    scale_pos_weight = len(pandas_Covid_einstein[pandas_Covid_einstein['SARS-Cov-2 exam result'] == 0]) / len(pandas_Covid_einstein[pandas_Covid_einstein['SARS-Cov-2 exam result'] == 1])

    param_grid = {'xgb__max_depth': [3, 4, 5, 6, 7, 8],
                  'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
                  'xgb__colsample_bytree': [0.6, 0.7, 0.8],
                  'xgb__min_child_weight': [0.4, 0.5, 0.6],
                  'xgb__gamma': [0, 0.01, 0.1],
                  'xgb__reg_lambda': [6, 7, 8, 9, 10],
                  'xgb__n_estimators': [150, 200, 300],
                  'xgb__scale_pos_weight': [scale_pos_weight]}

    rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=100,
                                n_jobs=-1, verbose=2, cv=5,                            
                                scoring='roc_auc', random_state=42)

    rs_clf.fit(x_train, y_train)
    
    print(f'XGBOOST BEST PARAMS: {rs_clf.best_params_}')
    
    y_pred = rs_clf.predict(x_test)

    df_results_xgb = pd.DataFrame(index=[['XGBoost'], ['No_sampler']], columns=['ROC_AUC', 'RECALL', 'ACCURACY'])

    df_results_xgb['ROC_AUC'] = roc_auc_score(y_test, y_pred)
    df_results_xgb['RECALL'] = recall_score(y_test, y_pred)
    df_results_xgb['ACCURACY'] = accuracy_score(y_test, y_pred)
    
    return df_results_xgb

# COMMAND ----------

df_results_xgb = train_xgb(pandas_Covid_einstein, xgb.XGBClassifier())

# COMMAND ----------

df_results = pd.concat([df_results_test, df_results_xgb])

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Evaluation des modèles 

# COMMAND ----------

df_plot = pd.concat([df_results.sort_values('ROC_AUC', ascending=False).head(3),
                     df_results.sort_values('RECALL', ascending=False).head(3),
                     df_results.sort_values('ACCURACY', ascending=False).head(3)])

# COMMAND ----------

def plot_test(pandas_Covid_einstein, xlim_min, xlim_max):

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,12))
    color = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'navy', 'turquoise', 'darkorange']

    pandas_Covid_einstein['ROC_AUC'].plot(kind='barh', ax=ax1, xlim=(xlim_min, xlim_max), title='ROC_AUC', color=color)
    pandas_Covid_einstein['RECALL'].plot(kind='barh', ax=ax2, xlim=(xlim_min, xlim_max), title='RECALL', color=color)
    pandas_Covid_einstein['ACCURACY'].plot(kind='barh', ax=ax3, xlim=(xlim_min, xlim_max), title='ACCURACY', color=color)
    plt.show()

# COMMAND ----------

plot_test(df_plot, 0.4, 1)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC *On peut constater que : 
# MAGIC 
# MAGIC - Pour le métrique : ROC-AUC 
# MAGIC 
# MAGIC 
# MAGIC      Le  modèle (Random Forest, Random_under_sampler) présentes de beaux scores d'accuracy pour la courbe d'apprentissage comme pour la courbe de validation. 
# MAGIC     Mais nous risquerons d'être  en presence d'un cas d'over-fitting. 
# MAGIC     
# MAGIC - Pour Recall (Rappel)
# MAGIC     
# MAGIC     (SVM, Random_over_sampler) a  une  valeur maximale de rappel et tend vers 1. 
# MAGIC 
# MAGIC 
# MAGIC - Pour l'accuracy 
# MAGIC     
# MAGIC   (XGBoost, No_sampler) a les meilleurs scores  
# MAGIC 
# MAGIC 
# MAGIC Nos meilleurs modèles sont les suivants:
# MAGIC 
# MAGIC     - Random Forest, Random_under_sampler
# MAGIC     
# MAGIC     - SVM, Random_over_sampler
# MAGIC     
# MAGIC     
# MAGIC     - XGBoost, No_sampler
# MAGIC    

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Analyse de matrices de confusions à différents seuils de confiance 

# COMMAND ----------

def plot_confusion_matrix(y_test, y_pred, title='Confusion matrix'):
    
    cm = confusion_matrix(y_test, y_pred)
    classes = ['No Covid', 'Covid']

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, )
    plt.title(title, fontsize=14)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def train_clf_threshold(pandas_Covid_einstein, clf, sampler=None):
    thresholds = np.arange(0.1, 1, 0.1)
    
    x_train, x_test, y_train, y_test = df_split(pandas_Covid_einstein)
    
    if sampler:
        clf_train = Pipeline([('sampler', sampler),
                              ('clf', clf)])
        
    else:        
        clf_train = clf
            
    clf_train.fit(x_train, y_train)
    y_proba = clf_train.predict_proba(x_test)
    
    plt.figure(figsize=(20,20))

    j = 1
    for i in thresholds:
        y_pred = y_proba[:,1] > i

        plt.subplot(4, 3, j)
        j += 1

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test,y_pred)
        np.set_printoptions(precision=2)

        print(f"Threshold: {round(i, 1)} | Test Accuracy: {round(accuracy_score(y_test, y_pred), 2)}| Test Recall: {round(recall_score(y_test, y_pred), 2)} | Test Roc Auc: {round(roc_auc_score(y_test, y_pred), 2)}")

        # Plot non-normalized confusion matrix
        plot_confusion_matrix(y_test, y_pred, title=f'Threshold >= {round(i, 1)}')

# COMMAND ----------

train_clf_threshold(pandas_Covid_einstein, RandomForestClassifier(), sampler=RandomUnderSampler())

# COMMAND ----------

## Seuil de confiance : 90%


TN = 1005
FP = 12
FN = 110
TP = 2

##  la sensibilité est le ratio du nombre de vrai positifs par le nombre total d'éléments positifs (y compris ceux déclarés faux par erreur).
## La spécificité c'est le ratio du nombre de vrai négatifs par le nombre total d'élément négatifs (y compris ceux déclarés vrai par erreur).

##

sensitivity = TP / float(FN + TP)

print("Sensibilité : %.2f" %  sensitivity)

specificity = TN / (TN + FP)

print("Spécificité  : %.2f" %  specificity)

precision = TP / float(TP + FP)

print("Précision  : %.2f" % precision)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Analyse de résultats 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 
# MAGIC - A partir de la matrice de confusion, de métriques d'évaluation ont été extraites pour permettre l'analyse du modèle. 
# MAGIC  
# MAGIC  - Le rappel ("recall"  en anglais), ou sensibilité ("sensitivity" en anglais), est le taux de vrais positifs, c’est à dire la proportion de positifs que l’on a correctement identifiés. C’est la capacité de notre modèle à détecter tous les patients décédés. 2% 
# MAGIC  
# MAGIC  - la précision, c’est-à-dire la proportion de prédictions correctes parmi les points que l’on a prédits positifs. C’est la capacité de notre modèle à ne déclencher le traitement que pour un vrai malade potentiellement risqué c'est à dire pouvant décéder. 14% 
# MAGIC  
# MAGIC  - La spécificité ("specificity" en anglais), qui est le taux de vrais négatifs, autrement dit la capacité à détecter toutes les  situations où il n’y a pas de décès. C’est une mesure complémentaire de la sensibilité. 99% 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### V.	CONCLUSION

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ####  Les résultats sont assez mitigé après ce premier entrainement. On peut noter les résultats :
# MAGIC - la classification  RandomForestClassifier de résultats encourangeants en terme de sensibilité, précision, et specificité. 
# MAGIC 
# MAGIC 
# MAGIC - le modèle AdaBoost pourrait être une alternative. 
# MAGIC 
# MAGIC 
# MAGIC #### Prochaines étapes pour améliorer le modèle:
# MAGIC 
# MAGIC 
# MAGIC - Ajouter de nouvelles variables sur la base de nouvelles ingestions de bases de données plus historiques sur les patients. 
# MAGIC 
# MAGIC - Création, transformation et génération des nouvelles variables plus discriminantes.
# MAGIC 
# MAGIC - Tunner les hyperparamètres pour diminuer l'overfitting. 
# MAGIC 
# MAGIC - Ajouter d'autres critères d'évaluation des performances des algorithmes. 
# MAGIC 
# MAGIC - Itérer plusieurs modèles en fonction des produits
