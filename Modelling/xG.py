# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 21:00:45 2021

@author: douil
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import FCPython 
import statsmodels.api as sm
import statsmodels.formula.api as smf

#On charge le maximum de championnats de façon à avoir une bonne estimation de l'expected goal avec notre modèle
with open('Wyscout/events/events_England.json') as f:
    data_England = json.load(f)
with open('Wyscout/events/events_European_Championship.json') as f:
    data_European_Championship = json.load(f)
with open('Wyscout/events/events_France.json') as f:
    data_France = json.load(f)
with open('Wyscout/events/events_Germany.json') as f:
    data_Germany = json.load(f)
with open('Wyscout/events/events_Italy.json') as f:
    data_Italy = json.load(f)
with open('Wyscout/events/events_Spain.json') as f:
    data_Spain = json.load(f)
with open('Wyscout/events/events_World_Cup.json') as f:
    data_World_Cup = json.load(f)

#Conversion en dataframe de tous les championnats
df_England = pd.DataFrame(data_England)
df_European_Championship = pd.DataFrame(data_European_Championship)
df_France = pd.DataFrame(data_France)
df_Germany = pd.DataFrame(data_Germany)
df_Italy = pd.DataFrame(data_Italy)
df_Spain = pd.DataFrame(data_Spain)
df_World_Cup = pd.DataFrame(data_World_Cup)

#Fusion des dataframes 
train = pd.concat([df_England, df_European_Championship, df_France, df_Germany, df_Italy, df_Spain, df_World_Cup], axis = 0)
pd.unique(train['subEventName'])
#On filtre sur le type d'event "Shot" pour construire l'xG
shots=train[train['subEventName']=='Shot']
shots_model=pd.DataFrame(columns=['Goal','X','Y'])

#On parcourt l'ensemble de notre dataframe shots pour ne garder que les coordonnées x, y
#On en profite également pour calculer en radian l'angle et en mètre la distance de chaque tir ainsi que s'ils correspondent à un but
for i,shot in shots.iterrows():
    shots_model.at[i,'X']=100-shot['positions'][0]['x']
    shots_model.at[i,'Y']=shot['positions'][0]['y']
    shots_model.at[i,'C']=abs(shot['positions'][0]['y']-50)

    x=shots_model.at[i,'X']*105/100
    y=shots_model.at[i,'C']*65/100
    shots_model.at[i,'Distance']=np.sqrt(x**2 + y**2)
    a = np.arctan(7.32 *x /(x**2 + y**2 - (7.32/2)**2))
    if a<0:
        a=np.pi+a
    shots_model.at[i,'Angle'] =a

    shots_model.at[i,'Goal']=0
    for shottags in shot['tags']:
            if shottags['id']==101:
                shots_model.at[i,'Goal']=1
                    
#Différentes variables ont été créé pour tester le modèle comme X², le produit de l'angle et de la distance...
squaredX = shots_model['X']**2
shots_model = shots_model.assign(X2=squaredX)
squaredC = shots_model['C']**2
shots_model = shots_model.assign(C2=squaredC)
AX = shots_model['Angle']*shots_model['X']
shots_model = shots_model.assign(AX=AX)
squaredD = shots_model['Distance']**2
shots_model = shots_model.assign(D2=squaredD)
AD = shots_model['Angle']*shots_model['Distance']
shots_model = shots_model.assign(AD=AD)

#Finalement on ne retiendra que l'angle et la distance
model_variables = ['Angle','Distance']
model=''
for v in model_variables[:-1]:
    model = model  + v + ' + '
model = model + model_variables[-1]

#Définition + fit du modèle en précisant la variable Goal comme cible
test_model = smf.glm(formula="Goal ~ " + model, data=shots_model, family=sm.families.Binomial()).fit()
print(test_model.summary())  
#Récupération des coefficients et de l'interception obtenus par le modèle      
b=test_model.params

#Calcul de l'xG à partir des coefficients obtenus par le modèle dans le but d'afficher un graphique qui représente l'xG sur un demi-terrain de football
def calculate_xG(sh):    
    """
    Cette fonction calcule l'xG à partir des coefficients déterminés par le modèle. 
    
    Paramètres :
        sh : dictionnaire contenant les variables qui ont été utilisé dans le modèle
        
    Renvoie :
        Le xG
    """
    
    bsum=b[0]
    for i,v in enumerate(model_variables):
        bsum=bsum+b[i+1]*sh[v]
    xG = 1/(1+np.exp(bsum)) 
    
    return xG   

xG=shots_model.apply(calculate_xG, axis=1) 
shots_model = shots_model.assign(xG=xG)

#Détermination de l'xG pour chaque position sur un demi-terrain
pgoal_2d=np.zeros((65,65))
for x in range(65):
    for y in range(65):
        sh=dict()
        a = np.arctan(7.32 *x /(x**2 + abs(y-65/2)**2 - (7.32/2)**2))
        if a<0:
            a = np.pi + a
        sh['Angle'] = a
        sh['Distance'] = np.sqrt(x**2 + abs(y-65/2)**2)
        sh['D2'] = x**2 + abs(y-65/2)**2
        sh['X'] = x
        sh['AX'] = x*a
        sh['AD'] = np.sqrt(x**2 + abs(y-65/2)**2)*a
        sh['X2'] = x**2
        sh['C'] = abs(y-65/2)
        sh['C2'] = (y-65/2)**2
        
        pgoal_2d[x,y] =  calculate_xG(sh)

#Affichage du graphique avec jauge de couleur pour mieux visualiser les variations
(fig,ax) = FCPython.createGoalMouth()
pos=ax.imshow(pgoal_2d, extent=[-1,65,65,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=0.3)
fig.colorbar(pos, ax=ax)
ax.set_title('Probability of goal')
plt.xlim((0,66))
plt.ylim((-3,35))
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
fig.savefig('Output/goalprobfor_' + model  + '.pdf', dpi=None, bbox_inches="tight")   


