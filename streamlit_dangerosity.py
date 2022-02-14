#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 15:30:09 2021

@author: alexandre.serrurier
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import seaborn as sns
from FCPython import createPitch
import matplotlib.pyplot as plt
import math
from IPython.display import display, clear_output
import statsmodels.api as sm
import statsmodels.formula.api as smf
import Metrica_Viz as mviz
import Metrica_IO as mio


wdir = "/Users/alexandre.serrurier/Documents/foot_datascientest"
mdir = '/Users/alexandre.serrurier/Documents/foot_datascientest/sample-data-master/data'

st.title('On Pytch Dangerosity')
st.write('Bienvenue sur cette interface qui va vous permettre de visualiser notre travail')


section = st.sidebar.selectbox(
    "Veuillez choisir la partie du projet à présenter",
    ("Introduction", "Partie 1 - Esprit du projet", "Partie 2 - Etude de séquences spécifiques")
)

if section == "Introduction":
    st.write("Ce projet a pour but d'essayer de modéliser la dangerosité d'une action de foot")
    st.write(""" L'objectif est d'essayer de donner une valeur au danger que représente une séquence de jeu. Au-delà des simples statistiques qu’on a l’habitude de voir (le nombre de tirs, la possession du ballon, les corners, etc), l'idée principale est d'apporer une couche supplémentaire qui permette de prendre également le contexte d'un match en compte. 
             Les métriques actuelles et également celles qui sont plus poussées (xG,xT) essayent de quantifier une performance mais ne s’intéressent qu’à la trajectoire de la balle au cours d’une action (joueurs directement concernés, actions,  ..) et donc isolent ce cheminement du reste du terrain. Or, le jeu sans ballon est très important car il permet de créer des décalages, des espaces ou un surnombre dans une zone précise.             
 
    Aucun de ces modèles ne prend en compte le contexte et l’apport des joueurs sans ballon, ils se contentent d’isoler une action et d’en tirer une conclusion sur l’apport individuel de joueurs. Nous essayons d'apporter une réponse à cela"""
 )
    st.subheader("Dans les figures ci-dessous vous verrez plus de précision sur l'objectif et l'esprit du projet")
    st.write("L'évolution de la dangerosité est une question de choix et cela peut mener à des mises en place de circuits préférentiels et de schémas de jeu")
    dang_1 = plt.imread(wdir + '/Dang_1.png')
    st.image(dang_1)
    dang_2 = plt.imread(wdir + '/Dang_2.png')
    st.image(dang_2)
    dang_3 = plt.imread(wdir + '/Dang_3.png')
    st.image(dang_3)
    
elif section == "Partie 1 - Esprit du projet":
    st.write("Dans cette section, nous allons nous plonger un peu plus dans le pourquoi de notre projet et montrer les data à notre disposition")
    st.caption("Cette partie se base sur le dataset Metrica")
    game = st.selectbox('Veuillez choisir un match',
                    ("-","1","2"))
    if game == "-":
        "Veuillez choisir un match pour continuer"

    else :
        events = mio.read_event_data(mdir,int(game))
        events = mio.to_metric_coordinates(events,field_dimen=(106.,68.))
        tracking_home = mio.tracking_data(mdir,int(game),'Home')
        tracking_away = mio.tracking_data(mdir,int(game),'Away')
        tracking_home = mio.to_metric_coordinates(tracking_home,field_dimen=(106.,68.))
        tracking_away = mio.to_metric_coordinates(tracking_away,field_dimen=(106.,68.))
      
        deb = st.text_input("Veuillez choisir le début de la séquence [mm:ss]")
        fin = st.text_input("Veuillez choisir la fin de la séquence [mm:ss]")
        
        deb_sec = int(deb.split(':')[1])
        deb_min = int(deb.split(':')[0])
        deb_tot = deb_min*60+deb_sec

        index_deb = int(deb_tot/0.04)

        fin_sec = int(fin.split(':')[1])
        fin_min = int(fin.split(':')[0])
        fin_tot = fin_min*60+fin_sec

        index_fin = int(fin_tot/0.04)
        if index_fin < index_deb:
            st.write("Veuillez choisir un début antérieur à la fin")
        else:
            st.caption('Voici à quoi ressemblent le dataframe des events')
            seq_events = events.iloc[deb_tot:fin_tot]
            df_tmp = seq_events.astype(str)
            st.dataframe(df_tmp)
            st.caption('Voici comment on peut exploiter les données de tracking')
            vid = mviz.save_match_clip(tracking_home.iloc[index_deb:index_fin],tracking_away.iloc[index_deb:index_fin],wdir,fname='vid',frames_per_second=50,include_player_velocities=False)
            video_file = open(wdir +'/vid.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            
elif section == "Partie 2 - Etude de séquences spécifiques":
    st.write("**_Identification des features importantes autour d'une séquence de jeu_** :soccer:")
    st.write("Dans cette partie, nous allons faire une modélisation calculée de la dangerosité d'une équipe en fonction de séquences précises")
    st.caption("Cette partie se base sur le dataset Metrica")
    game = st.selectbox('Veuillez choisir un match',
                    ("-","1","2"))

    if game == "-":
        "Veuillez choisir un match pour continuer"

    else :
        events = mio.read_event_data(mdir,int(game))
        events = mio.to_metric_coordinates(events,field_dimen=(106.,68.))
        tracking_home = mio.tracking_data(mdir,int(game),'Home')
        tracking_away = mio.tracking_data(mdir,int(game),'Away')
        tracking_home = mio.to_metric_coordinates(tracking_home,field_dimen=(106.,68.))
        tracking_home = tracking_home.drop(['Period','Time [s]'],axis = 1)
        tracking_away = mio.to_metric_coordinates(tracking_away,field_dimen=(106.,68.))
        tracking_away = tracking_away.drop(['Period','Time [s]'],axis = 1)
    
        events['Equipe'] = events['Team'].apply(lambda x : 1 if x == 'Home' else 0)
        events['Sequence'] = ""
        events['Sequence'][0] = 1
        for i in np.arange(1,events.shape[0],1):
            if events['Equipe'][i] == events['Equipe'][i-1]:
                events['Sequence'][i] = events['Sequence'][i-1]
            else:
                events['Sequence'][i] = events['Sequence'][i-1] + 1
        seq_gr = events.groupby(['Team','Sequence'],as_index=False)
        st.text('Il y a en tout ' + str(len(seq_gr)) + ' séquences dans ce match')
    
        seq = st.slider('Veuillez choisir une séquence du match',1,len(seq_gr))
        df_seq = events[events['Sequence'] == seq]
        df_seq = df_seq.reset_index(inplace = False, drop = True)
        timeline = np.arange(0,df_seq.shape[0])
        
   
# Détermination de la direction de jeu
        period = df_seq['Period'][0] 
        team = df_seq['Team'][0]
        if team == 'Home':
            tracking_dir = tracking_home
            elt = tracking_dir[tracking_dir.columns[0]][1]
            if period == 1:
                direction = -np.sign(elt)
            else : 
                direction = np.sign(elt)
        else :
            tracking_dir = tracking_away
            elt = tracking_dir[tracking_dir.columns[0]][1]
            if period == 1:
                direction = -np.sign(elt)
            else : 
                direction = np.sign(elt)

        if df_seq['Team'][0] == 'Home':
            tracking = tracking_away
        else : 
            tracking = tracking_home
    
        df_seq['Timeline'] = np.arange(0,df_seq.shape[0])    
        df_seq['xG'] = ""
        df_seq['distPlay'] = ""
        df_seq['oppLeft'] = ""
        df_seq['dang'] = ""
    
# Calcul des éléments de la dangerosité 
      
        for i in timeline:
            timeframe = df_seq['Start Frame'][i]  
            players = tracking.iloc[timeframe,:]
            
## Calcul du xG
            x2 = -direction*df_seq['Start X'][i]
            y2 = df_seq['Start Y'][i]
            if math.isnan(x2) or math.isnan(y2):
                df_seq['xG'][i] = 0
            else :
                distance = np.sqrt(x2**2 + y2**2)
                a = np.arctan(7.32 * x2 / (x2**2 + y2**2 - (7.32 / 2)**2))
                if a < 0:
                    a = np.pi + a
                angle = a * 180 / np.pi
                df_seq['xG'][i] = round(1 / (1 + np.exp(1.5472 - 1.4622 * a + 0.0871 * distance)),4)
    
## Détermination du joueur le plus proche
            coeff_dir = (df_seq['End Y'][i] - df_seq['Start Y'][i]) / (df_seq['End X'][i] - df_seq['Start X'][i])
            ordOrigin = - coeff_dir * df_seq['End X'][i] + df_seq['End Y'][i]
            distance = []
            for j in np.arange(0,int(len(players)),2):
                x_p = players[j]
                y_p = players[j+1]
                distanceOpp = abs(coeff_dir * x_p - y_p + ordOrigin) / math.sqrt(coeff_dir**2 + 1)
                distance.append(distanceOpp)
                dmin = min(distance)
                if math.isnan(dmin):
                    dmin = 2
                df_seq['distPlay'][i] = round(dmin,4)
    

## Détermination du nombre de joueurs restants avant le but
 #On va parcourir l'ensemble des joueurs adverses et déterminer ceux qui ont une abscisse supérieure à celle de l'attaquant  
            vi_oppLeft = 0
            for j in np.arange(0,int(len(players)),2):
                x_p = players[j]
                if direction == 1:
                    if x_p > df_seq['Start X'][i]:
                        vi_oppLeft += 1
                    else : 
                        vi_oppLeft = vi_oppLeft
                else :
                    if x_p < df_seq['Start X'][i]:
                        vi_oppLeft += 1
                    else : 
                        vi_oppLeft = vi_oppLeft
                df_seq['oppLeft'][i] = vi_oppLeft

    
        df_seq['dang'] = 100*(df_seq['xG']*0.1*df_seq['distPlay']/(1+df_seq['oppLeft']))
        df_tmp = df_seq.astype(str)
        st.dataframe(df_tmp)

    
        graph = st.selectbox(
            'Veuillez choisir la visualisation que vous souhaitez',
            ('Terrain', 'Timeline'))

        if graph == "Terrain":
            fig,ax = mviz.plot_pitch()
            ax.scatter(df_seq['Start X'],df_seq['Start Y'],s = 100000*df_seq['dang'].astype(float), c = df_seq['Timeline'],cmap = 'Blues')
            ax.arrow(0,35,20*direction,0,fc = 'k',width = 0.5)
            for i in timeline:
                ax.plot([df_seq['Start X'][i],df_seq['End X'][i]], [df_seq['Start Y'][i],df_seq['End Y'][i]])
                ax.text(df_seq['Start X'][i]+0.5,df_seq['Start Y'][i],df_seq['Type'][i],horizontalalignment = 'left', size = 'small', color = 'black')
            for i in np.arange(0,len(timeline)-1,1):
                ax.plot([df_seq['End X'][i],df_seq['Start X'][i+1]], [df_seq['End Y'][i],df_seq['Start Y'][i+1]],linestyle = '--',color = 'g')
            ax.set_xlabel('position x')
            ax.set_ylabel('position y')
            st.pyplot(fig)
    
        else :
            fig,ax = plt.subplots()
            ax.plot(df_seq['dang'],label = 'Dangerosité')
            ax.plot(df_seq['xG'],label = 'xG')
            ax.plot(df_seq['distPlay'],label = 'Distance adversaire')
            ax.plot(df_seq['oppLeft']/10,label = "Nb adversaires")
            ax.legend()
            
            st.pyplot(fig)






    