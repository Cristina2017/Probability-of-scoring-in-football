# -*- coding: utf-8 -*-
"""
Created on Sat May 22 19:35:12 2021

@author: douil
"""

import json
import pandas as pd

#On charge l'ensmble des matches de SkillCorner
#C:\Users\crist\OneDrive\Documents\Data Scientest\Proyecto\friends of tracking
with open('C:/Users/crist/OneDrive/Documents/Data Scientest/Proyecto/matches/matches.json') as f:
    matches = json.load(f)

#On défini un dictionnaire qui contiendra l'ID du match, le nom des deux équipes
#Le nombre toal d'observations, et parmi elles, celles qui sont comprises dans le temps de jeu et celles contenant des données
dict_matches = {'ID': [],
                'Match': [],
                'Total': [],
                'In game': [],
                'With data': []}

#On parcourt chaque match
for match in matches:
    #Pour chaque match on stocke ses données de tracking et ses meta données
    with open('C:/Users/crist/OneDrive/Documents/Data Scientest/Proyecto/matches/' + str(match["id"]) + '/structured_data.json') as g:
        structured_data = json.load(g)
    with open('C:/Users/crist/OneDrive/Documents/Data Scientest/Proyecto/matches/' + str(match["id"]) + '/match_data.json') as h:
        match_data = json.load(h)
    #On alimente notre dictionnaire qui sera la source de données de nos graphiques
    dict_matches["ID"].append(match['id'])
    dict_matches["Match"].append(match_data['home_team']['acronym'] + " - " + match_data['away_team']['acronym'])
    dict_matches["Total"].append(len(structured_data))
    #Pour les observations "in game" on vérifie simplement qu'elles ont bien une durée du match
    #Pour les observations contenant des données on vérifie que le champ "data" n'est pas vide
    nbObsInGame = 0
    nbObsInGameWithData = 0
    for data in structured_data:
        if data["period"] != None:
            nbObsInGame += 1
            if len(data["data"]) != 0:
                nbObsInGameWithData += 1
    dict_matches["In game"].append(nbObsInGame)
    dict_matches["With data"].append(nbObsInGameWithData)

#On affiche le graphique
df = pd.DataFrame(data = dict_matches, index = dict_matches["ID"])
df.plot.bar(x = "Match", y = ["Total", "In game", "With data"], rot = 0, figsize = (12, 10), title = "Observations by match")