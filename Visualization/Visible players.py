# -*- coding: utf-8 -*-
"""
Created on Sat May 22 19:35:12 2021

@author: douil
"""

import json
import pandas as pd

#Ouverture du fichier format json contenant les 9 matches SkillCorner public
with open('opendata-master/data/matches.json') as f:
    matches = json.load(f)

#On déclare une variable qui stocke la valeur en dixième de secondes de la période de calcul
#De cette façon on "lisse" les graphiques de manière à avoir une meilleure lisibilité
dureeLissage = 600

#On parcourt l'ensemble des matches SkillCorner
for match in matches:
    #On construit un dictionnaire qui va contenir pour chaque frame le nombre de joueurs identifiés et le nombre de joueurs total
    dict_matches = {'frame' : [],
                    'trackedPlayers': [],
                    'identifiedPlayers': []}
    #on ouvre les données de tracking et les meta données de chaque match grâce à leur identifiant SkillCorner
    with open('opendata-master/data/matches/' + str(match["id"]) + '/structured_data.json') as g:
        structured_data = json.load(g)
    with open('opendata-master/data/matches/' + str(match["id"]) + '/match_data.json') as h:
        match_data = json.load(h)
    #on stocke dans une liste les id des objets qui ne sont pas des joueurs de manière à les filtrer dans les graphiques
    #on n'affichera pas le ballon et les arbitres
    nonPlayerObject = []
    i = 0
    nonPlayerObject.append(match_data['ball']["trackable_object"])
    for referee in match_data['referees']:
        nonPlayerObject.append(referee["trackable_object"])
    identifiedPlayers = 0
    unidentifiedPlayers = 0
    #on parcourt l'ensmble des données de tracking
    for data in structured_data:
        for trackedObject in data["data"]:
            #si le champ "group_name" apparaît dans le tracking de l'objet c'est que la computer vision n'a pas réussi à l'identifier
            if "group_name" in trackedObject:
                unidentifiedPlayers += 1
            else:
                if trackedObject['trackable_object'] not in nonPlayerObject:
                    identifiedPlayers += 1
        totalTrackedPlayers = unidentifiedPlayers + identifiedPlayers
        #on n'alimente pas le dictionnaire pour chaque, sinon la courbe serait "illisible" car trop sujette aux variations
        #à la place on calcule une moyenne sur la période de "lissage" définie plus haut
        if i % dureeLissage == 0:
            dict_matches['frame'].append(data['frame'])
            dict_matches['trackedPlayers'].append(totalTrackedPlayers / dureeLissage)
            dict_matches['identifiedPlayers'].append(identifiedPlayers / dureeLissage)
            identifiedPlayers = 0
            unidentifiedPlayers = 0
        i += 1
    #on affiche le graphique 
    df = pd.DataFrame(data = dict_matches, index = dict_matches["frame"])
    df.plot(x = "frame", y = ["trackedPlayers", "identifiedPlayers"], figsize = (20, 5), title = "Tracked players " + match_data['home_team']['acronym'] + " - " + match_data['away_team']['acronym'] + " (avg for 1 min)")
