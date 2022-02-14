import numpy as np
import pandas as pd
import csv as csv
import json
import math
import scipy.signal as signal
import matplotlib.pyplot as plt

# In this first step we are going to introduce the functions that allows us to work with data  

def read_match_data(DATADIR,gameid):
    '''
    read_match_data(DATADIR,gameid):
    read all Metrica match data (tracking data for home & away teams, and ecvent data)
    '''
    tracking_home = tracking_data(DATADIR,gameid,'Home')
    tracking_away = tracking_data(DATADIR,gameid,'Away')
    events = read_event_data(DATADIR,gameid)
    return tracking_home,tracking_away,events

def read_event_data(DATADIR,game_id):
    '''
    read_event_data(DATADIR,game_id):
    read Metrica event data  for game_id and return as a DataFrame
    '''
    eventfile = '/Sample_Game_%d/Sample_Game_%d_RawEventsData.csv' % (game_id,game_id) # filename
    events = pd.read_csv('{}/{}'.format(DATADIR, eventfile)) # read data
    return events

def tracking_data(DATADIR,game_id,teamname):
    '''
    tracking_data(DATADIR,game_id,teamname):
    read Metrica tracking data for game_id and return as a DataFrame. 
    teamname is the name of the team in the filename. For the sample data this is either 'Home' or 'Away'.
    '''
    teamfile = '/Sample_Game_%d/Sample_Game_%d_RawTrackingData_%s_Team.csv' % (game_id,game_id,teamname)
    # First:  deal with file headers so that we can get the player names correct
    csvfile =  open('{}/{}'.format(DATADIR, teamfile), 'r') # create a csv file reader
    reader = csv.reader(csvfile) 
    teamnamefull = next(reader)[3].lower()
    print("Reading team: %s" % teamnamefull)
    # construct column names
    jerseys = [x for x in next(reader) if x != ''] # extract player jersey numbers from second row
    columns = next(reader)
    for i, j in enumerate(jerseys): # create x & y position column headers for each player
        columns[i*2+3] = "{}_{}_x".format(teamname, j)
        columns[i*2+4] = "{}_{}_y".format(teamname, j)
    columns[-2] = "ball_x" # column headers for the x & y positions of the ball
    columns[-1] = "ball_y"
    # Second: read in tracking data and place into pandas Dataframe
    tracking = pd.read_csv('{}/{}'.format(DATADIR, teamfile), names=columns, index_col='Frame', skiprows=3)
    return tracking

def merge_tracking_data(home,away):
    '''
    merge home & away tracking data files into single data frame
    '''
    return home.drop(columns=['ball_x', 'ball_y']).merge( away, left_index=True, right_index=True )

def merge_all(tracking, events):
    return pd.merge(events, tracking, left_on='Start Frame', right_on='Frame')
    
def to_metric_coordinates(data,field_dimen=(106.,68.) ):
    '''
    Convert positions from Metrica units to meters (with origin at centre circle)
    '''
    x_columns = [c for c in data.columns if c[-1].lower()=='x']
    y_columns = [c for c in data.columns if c[-1].lower()=='y']
    data[x_columns] = ( data[x_columns]-0.5 ) * field_dimen[0]
    data[y_columns] = -1 * ( data[y_columns]-0.5 ) * field_dimen[1]
    ''' 
    ------------ ***NOTE*** ------------
    Metrica actually define the origin at the *top*-left of the field, not the bottom-left, as discussed in the YouTube video. 
    I've changed the line above to reflect this. It was originally:
    data[y_columns] = ( data[y_columns]-0.5 ) * field_dimen[1]
    ------------ ********** ------------
    '''
    return data

def calc_player_velocities(team, maxspeed = 12):
    
    '''
    We are going to get the velocity of all players
    '''
     # Get the player ids
    player_ids = np.unique( [ c[:-2] for c in team.columns if c[:4] in ['Home','Away'] ] )

    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = team['Time [s]'].diff()
    
    # index of first frame in second half
    second_half_idx = team.Period.idxmax(2)
    
    # estimate velocities for players in team
    for player in player_ids:
        vx = team[player+"_x"].diff() / dt
        vy = team[player+"_y"].diff() / dt

        if maxspeed>0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_speed = np.sqrt( vx**2 + vy**2 )
            vx[ raw_speed>maxspeed ] = np.nan
            vy[ raw_speed>maxspeed ] = np.nan
            
        team[player + "_vx"] = vx
        team[player + "_vy"] = vy
        team[player + "_speed"] = np.sqrt( vx**2 + vy**2 )

    return team


def calc_attaq_def(total):
    '''
    We are going to identify the players that are in a 4 metres radium circle, considering the center of the circle the player
    who has the ball.
    
    '''   
    centre_circle_radius = 4   
    player_home = np.unique([c.split('_')[1] for c in total.columns if c[:4] in 'Home'])
    player_away = np.unique([c.split('_')[1] for c in total.columns if c[:4] in 'Away'])
    for i, row in total.iterrows():
        try:
            defense = []
            attaque = []
            porteur_ball = row['From'][6:]
            if porteur_ball in player_home:
                for player in player_home:
                    
                    if np.sqrt((row["Home"+"_"+player+"_" + "x"] - row['Home'+"_"+porteur_ball+"_"+"x"])**2+
                               (row["Home"+"_"+player+"_" + "y"]- row['Home'+"_"+porteur_ball+"_"+"y"])**2)<= centre_circle_radius:
                        attaque.append(player)
                        total.at[i,'Attaque'] = attaque
              
                for player in player_away:
                    if np.sqrt((row["Away"+"_"+player+"_" + "x"] - row['Home'+"_"+porteur_ball+"_"+"x"])**2+
                               (row["Away"+"_"+player+"_" + "y"]- row['Home'+"_"+porteur_ball+"_"+"y"])**2)<= centre_circle_radius:
                        defense.append(player)    
                        total.at[i,'Defense'] = defense
            else:
                for player in player_home:
                    if np.sqrt((row["Home"+"_"+player+"_" + "x"] - row['Away'+"_"+porteur_ball+"_"+"x"])**2+
                               (row["Home"+"_"+player+"_" + "y"]- row['Away'+"_"+porteur_ball+"_"+"y"])**2)<= centre_circle_radius:
                        defense.append(player)
                        total.at[i,'Defense'] = defense
                for player in player_away:
                    if np.sqrt((row["Away"+"_"+player+"_" + "x"] - row['Away'+"_"+porteur_ball+"_"+"x"])**2+
                               (row["Away"+"_"+player+"_" + "y"]- row['Away'+"_"+porteur_ball+"_"+"y"])**2)<= centre_circle_radius:
                        attaque.append(player)    
                        total.at[i,'Attaque'] = attaque
        except:
            pass
    return total

def score_attaque(total):
    '''
    Knowing the attacking players, we are going to assign an attack score, considering the speed and direction of the player
    
    '''
    for i, row in total.iterrows():
        try:
            score = []
            porteur_ball = row['From'][6:]
            porteur_ball_equipe = row['Team']
            porteur_x = row[ porteur_ball_equipe+ "_" + porteur_ball + "_" + "x"]
            porteur_y = row[ porteur_ball_equipe+ "_" + porteur_ball + "_" + "y"]
            
            if isinstance(row["Attaque"], list):
                for player in row['Attaque']:
                    if porteur_ball_equipe == "Home":
                        equipe = "Home"
                    else:
                        equipe = "Away"
                    if player != porteur_ball:
                        Jx = row[equipe +"_" + player + "_" + "x"]
                        Jy = row[equipe + "_" + player + "_" + "y"]
                        vx = row[equipe + "_" + player + "_" + "vx"]
                        vy = row[equipe + "_" + player + "_" + "vx"]
                    # S'il y a un erreur avec les données et pour deux jouers les coordonnées sont pareilles.(trouvé pour la defense dans le game 2)
                        if Jx == porteur_x or Jy == porteur_y:
                            Jx = (total.at[i-1, equipe + "_" + player + "_" + "x"] + total.at[i-2, equipe +"_"+ player +"_"+"x"])/2
                            Jy = (total.at[i-1, equipe +"_"+ player +"_"+"y"] + total.at[i-2, equipe+"_"+player+"_"+"y"])/2
                            vx = (total.at[i-1, equipe + "_" + player + "_" + "vx"] + total.at[i-2, equipe +"_"+ player +"_"+"vx"])/2
                            vy = (total.at[i-1, equipe + "_" + player + "_" + "vy"] + total.at[i-2, equipe +"_"+ player +"_"+"vy"])/2
                            s = ((vx*(Jx-porteur_x)+vy*(Jy-porteur_y))/((Jx-porteur_x)**2 + (Jy-porteur_y)**2))
                            score.append(s)     
            solution = abs(round(np.sum(score),4))  
            total.at[i,'score_attaque'] = solution
        except:
            pass
    return total

def score_defense(total):
    '''
    Knowing the defenders players, we are going to assign a defense score, considering the speed and direction of the player
    
    '''
    for i, row in total.iterrows():
        try:
            score = []
            porteur_ball = row['From'][6:]
            porteur_ball_equipe = row['Team']
            porteur_x = row[ porteur_ball_equipe+ "_" + porteur_ball + "_" + "x"]
            porteur_y = row[ porteur_ball_equipe+ "_" + porteur_ball + "_" + "y"]
            if porteur_ball_equipe == "Home":
                equipe = "Away"
            else:
                equipe = "Home"
            if isinstance(row["Defense"], list):
                for player in row['Defense']:
                    Jx = row[equipe +"_" + player + "_" + "x"]
                    Jy = row[equipe + "_" + player + "_" + "y"]
                    vx = row[equipe + "_" + player + "_" + "vx"]
                    vy = row[equipe + "_" + player + "_" + "vy"]
                    if Jx == porteur_x or Jy == porteur_y:
                        Jx = (total.at[i-1, equipe + "_" + player + "_" + "x"] + total.at[i-2, equipe +"_"+ player +"_"+"x"])/2
                        Jy = (total.at[i-1, equipe +"_"+ player +"_"+"y"] + total.at[i-2, equipe+"_"+player+"_"+"y"])/2
                        vx = (total.at[i-1, equipe + "_" + player + "_" + "vx"] + total.at[i-2, equipe +"_"+ player +"_"+"vx"])/2
                        vy = (total.at[i-1, equipe + "_" + player + "_" + "vy"] + total.at[i-2, equipe +"_"+ player +"_"+"vy"])/2
                    s = ((vx*(Jx-porteur_x)+vy*(Jy-porteur_y))/((Jx-porteur_x)**2 + (Jy-porteur_y)**2))
                    score.append(s)
            solution = abs(round(np.sum(score), 4))
            total.at[i,'score_defense'] = solution 
        except:
            pass
    return total

def calculate_xG(x, y, period, team, direction):
    '''
     We are going to calculate the xG. Taking into consideration the following parameters 
    
     x : x-axis
     y : y-axis
     period : first or second half of the match
     team : team with the ball
     direction : attaque direction
        
    The result of this function is the adversary's goal distance, shooting angle and xG
        
    '''
    # To avoid outside coodinates we are going to limitate the extreme values
    if x >= 1:
        x = .99
    elif x <= 0:
        x = 0.01
    if y >= 1:
        y = .99
    elif y <= 0:
        y = 0.01
    
    if direction == -1:
        if (period == 1 and (team == "Home" or team == "Team A")) or (period == 2 and (team == "Away" or team == "Team B")):
            x = 1 - x
    else:
        if (period == 1 and (team == "Away" or team == "Team B")) or (period == 2 and (team == "Home" or team == "Team A")):
            x = 1 - x
    
    x2 = (1 - x) * 105
    y2 = abs(y - 0.5) * 68
    distance = np.sqrt(x2**2 + y2**2)
    a = np.arctan(7.32 * x2 / (x2**2 + y2**2 - (7.32 / 2)**2))
    if a < 0:
        a = np.pi + a
    
    angle = a * 180 / np.pi
    xG = 1 / (1 + np.exp(1.5472 - 1.4622 * a + 0.0871 * distance))
    
    return distance, angle, xG

# We execute and apply the functions
    
DATADIR = 'C:/Users/crist/OneDrive/Documents/Data Scientest/Proyecto/metrica'

events1 = read_event_data(DATADIR,1)

# read in tracking data
tracking_home1 = tracking_data(DATADIR,1,'Home')
tracking_away1 = tracking_data(DATADIR,1,'Away')

# We add the xG:

direction = -1
for c in tracking_home1.columns:
    if "_x" in c and "ball" not in c:
        if not np.isnan(tracking_home1.iloc[events1.iloc[0]['End Frame'] - 1][c]):
            if tracking_home1.iloc[events1.iloc[0]['End Frame'] - 1][c] < 0.4:
                direction = 1
for i, row in events1.iterrows():
        if not np.isnan(row['End X']) and not np.isnan(row['End Y']):
            events1.at[i,'xG'] = calculate_xG(row['End X'], row['End Y'], row['Period'], row['Team'], direction)[2]                   

# As we have the coordinates in metrica units we are going to transform it into meters

tracking_home1 = to_metric_coordinates(tracking_home1)
tracking_away1 = to_metric_coordinates(tracking_away1)
tracking_home1 = calc_player_velocities(tracking_home1)
tracking_away1 = calc_player_velocities(tracking_away1)
tracking = merge_tracking_data(tracking_home1,tracking_away1)
total1 = merge_all(tracking, events1)

events2 = read_event_data(DATADIR,2)
tracking_home2 = tracking_data(DATADIR,2,'Home')
tracking_away2 = tracking_data(DATADIR,2,'Away') 


direction = -1
for c in tracking_home2.columns:
    if "_x" in c and "ball" not in c:
        if not np.isnan(tracking_home2.iloc[events2.iloc[0]['End Frame'] - 1][c]):
            if tracking_home2.iloc[events2.iloc[0]['End Frame'] - 1][c] < 0.4:
                direction = 1
for i, row in events2.iterrows():
        if not np.isnan(row['End X']) and not np.isnan(row['End Y']):
            events2.at[i,'xG'] = calculate_xG(row['End X'], row['End Y'], row['Period'], row['Team'], direction)[2]                   

tracking_home2 = to_metric_coordinates(tracking_home2)
tracking_away2 = to_metric_coordinates(tracking_away2)
tracking_home2 = calc_player_velocities(tracking_home2)
tracking_away2 = calc_player_velocities(tracking_away2)
tracking2 = merge_tracking_data(tracking_home2,tracking_away2)
total2 = merge_all(tracking2, events2)

# Game 3 is formatted differently, so we build a dataframe from raw data.

with open(DATADIR + '/Sample_Game_3/Sample_Game_3_events.json') as f:
    Game_3 = json.load(f)

tracking3 = pd.read_csv(DATADIR + '/Sample_Game_3/Sample_Game_3_tracking.txt', sep="[,;:]", engine="python", header = None, index_col=False)


tracking3_home_columns_x = ["Home"+ "_"+ str(i) + "_" + "x" for i in range(1,11)]
tracking3_home_columns_x.insert(0, "Home_11_x")
tracking3_home_columns_y = ["Home"+ "_"+ str(i) + "_" + "y" for i in range(1,11)]
tracking3_home_columns_y.insert(0, "Home_11_y")

tracking3_away_columns_x = ["Away"+ "_"+ str(i) + "_" + "x" for i in range(12,23)]
tracking3_away_columns_x.insert(0, "Away_23_x")
tracking3_away_columns_y = ["Away"+ "_"+ str(i) + "_" + "y" for i in range(12,23)]
tracking3_away_columns_y.insert(0, "Away_23_y")

home_columns = list(zip(tracking3_home_columns_x, tracking3_home_columns_y))
home_columns = [c for e in home_columns for c in e]

away_columns = list(zip(tracking3_away_columns_x, tracking3_away_columns_y))
away_columns = [c for e in away_columns for c in e]

tracking3_columns = home_columns + away_columns
tracking3_columns.insert(0,'Frame')
tracking3.columns = tracking3_columns


tracking3.insert(1, 'Period', float(np.nan))

p1=[]
for i in range(69661):
    p1.append(1)
p2=[]
for i in range(69662, 143762):
    p2.append(2)
period = p1+p2
tracking3['Period'] = period

time = np.arange(0.04,5750.48,0.04)
tracking3.insert(2, 'Time [s]',time )


# Events: we take the information in series to built a dataframe

team = []
for element in Game_3['data']:
    if element is not None:
        t = element['team']['name']
        team.append(t)
    else:
        team.append(None)
        
Type = []
for element in Game_3['data']:
    if element is not None:
        tp = element['type']['name']
        Type.append(tp)
    else:
        Type.append(None)
        
subtypes = []
for element in Game_3['data']:
    subtypes.append(element['subtypes'])
Subtype = []
for element in subtypes:
    if element is not None:
        if isinstance(element, list): 
            Subtype.append(element[0]['name'] + " "+ element[1]['name'])          
        else:
            Subtype.append(element['name'])            
    else:
        Subtype.append(None)
        

Period = []
for element in Game_3['data']:
    if element is not None:
        per = element['period']
        Period.append(per)
    else:
        Period.append(None)

From = []
for element in Game_3['data']:
    if element is not None:
        f = element['from']['name']
        From.append(f)
    else:
        From.append(None)
        
To = []
for element in Game_3['data']:
    if element is not None:
        t = element['to']
        if t is not None:
            To.append(t['name'])
        else:
            To.append(None)
            
Start_frame = []
for element in Game_3['data']:
    if element is not None:
        data_start = element['start']
        if data_start is not None:
            frame = data_start['frame']
            if frame is not None:
                Start_frame.append(frame)
            else:
                Start_frame.append(None)
                
Start_time = []
for element in Game_3['data']:
    if element is not None:
        data_start = element['start']
        if data_start is not None:
            s_time = data_start['time']
            if s_time is not None:
                Start_time.append(s_time)
            else:
                Start_time.append(None)

Start_x = []
for element in Game_3['data']:
    if element is not None:
        data_start = element['start']
        if data_start is not None:
            x = data_start['x']
            if x is not None:
                Start_x.append(x)
            else:
                Start_x.append(None)
                
Start_y = []
for element in Game_3['data']:
    if element is not None:
        data_start = element['start']
        if data_start is not None:
            y = data_start['y']
            if y is not None:
                Start_y.append(y)
            else:
                Start_y.append(None)
                
End_frame = []
for element in Game_3['data']:
    if element is not None:
        data_end = element['end']
        if data_end is not None:
            frame_end = data_end['frame']
            if frame_end is not None:
                End_frame.append(frame_end)
            else:
                End_frame.append(None)
                
End_Time = []
for element in Game_3['data']:
    if element is not None:
        data_end = element['end']
        if data_end is not None:
            time_end = data_end['time']
            if time_end is not None:
                End_Time.append(time_end)
            else:
                End_Time.append(None)
                
End_x = []
for element in Game_3['data']:
    if element is not None:
        data_end = element['end']
        if data_end is not None:
            x_end = data_end['x']
            if x_end is not None:
                End_x.append(x_end)
            else:
                End_x.append(None)
 

End_y = []
for element in Game_3['data']:
    if element is not None:
        data_end = element['end']
        if data_end is not None:
            y_end = data_end['x']
            if y_end is not None:
                End_y.append(y_end)
            else:
                End_y.append(None)               

d = {'Team':team, 'Type': Type, 'Subtype': Subtype, 'Period':Period, 'Start Frame':Start_frame, 'Start Time [s] ': Start_time, 'End Frame': End_frame, 'End Time [s]': End_Time, 'From':From, 'To':To, 'Start X': Start_x, 'Start Y': Start_y, 'End X': End_x, 'End Y':End_y}
events3 = pd.DataFrame(d)
events3['Team']=events3['Team'].replace(to_replace=["Team A", 'Team B'], value=['Home', 'Away'])
events3['From'] = events3['From'].apply(lambda x: x.replace(" ", ""))
for i, row in events3.iterrows():
    if row['To'] is not None:
        events3.at[i,'To'] = events3.at[i,'To'].replace(" ", "")

events3['To']=events3['To'].replace(to_replace=["Player29", "Player30", "Player31", "Player32", "Player33", "Player34", "Player35"], value=[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])        
direction = -1
for c in tracking3.columns:
    if "_x" in c and "ball" not in c:
        if not np.isnan(tracking3.iloc[events3.iloc[0]['End Frame'] - 1][c]):
            if tracking3.iloc[events3.iloc[0]['End Frame'] - 1][c] < 0.4:
                direction = 1            

             
for i, row in events3.iterrows():
        if not np.isnan(row['End X']) and not np.isnan(row['End Y']):
            events3.at[i,'xG'] = calculate_xG(row['End X'], row['End Y'], row['Period'], row['Team'], direction)[2] 
tracking3 = to_metric_coordinates(tracking3)
tracking3 = calc_player_velocities(tracking3)
total3 = merge_all(tracking3, events3)
total3 = total3.rename({'Period_x':'Period'}, axis=1)

total = pd.concat([total1, total2, total3], ignore_index=True, axis = 0)

total = total.drop(['Time [s]_x', 'Time [s]_y', "Period_x", "Period_y"], axis = 1)
total = total.rename(columns={'End Time [s]':'End Time','Start Time [s]': 'Start Time' }) 
total['Start Time'] = list(total['Start Time'][:3680])+Start_time

# For match 2 player26 have an space between player and the number, we are going to remove it.

total['From']=total['From'].replace(to_replace=["Player 26"], value=['Player26'])
total['To']=total['To'].replace(to_replace=["Player 26"], value=['Player26'])

total = calc_attaq_def(total)
total = score_attaque(total)
total = score_defense(total)   
        
# As we are interested in interception, we are going to take only the types pass and challenge    

inter_model = total[(total['Type']== "PASS") | (total['Type']== "CHALLENGE")]
ind = np.arange(0,len(inter_model), 1)
inter_model['index'] = ind
inter_model = inter_model.set_index('index') 
inter_model = inter_model.drop(['Start Time [s] ', 'Frame', 'Time [s]'], axis=1)

inter_model['Vitesse ball'] = float('nan')   
inter_model['joueur_plus_proche'] = float('nan')
inter_model['position_jou_proche_x'] = float('nan')
inter_model['position_jou_proche_y'] = float('nan')  
inter_model['vitesse_jou_proche'] = float('nan')
inter_model['position_porteur_x'] = float('nan')
inter_model['position_porteur_y'] = float('nan')
inter_model['vitesse_porteur'] = float('nan')
inter_model['position_recepteur_x'] = float('nan')
inter_model['position_recepteur_y'] = float('nan')
inter_model['vitesse_recepteur'] = float('nan')
inter_model['dist_jou_plus_proche'] = float('nan')
inter_model['dist_port_jou'] = float('nan')
       
for i, row in inter_model.iterrows():
    # ball's speed:
    
    distance = np.sqrt((row['End Y'] - row['Start Y'])**2+(row['End X'] - row['Start X'])**2)
    t = row['End Time'] - row['Start Time']
    Vit_ball = distance/t
    inter_model.at[i, 'Vitesse ball'] = Vit_ball
    
    # Ball's trajectory:

    player_home = np.unique([c.split('_')[1] for c in total.columns if c[:4] in 'Home'])
    player_away = np.unique([c.split('_')[1] for c in total.columns if c[:4] in 'Away'])
    porteur_ball = row['Team']
    if porteur_ball == 'Home':
        equipe = "Away"
    else:
        equipe = "Home"
    players = np.unique([c.split('_')[1] for c in inter_model.columns if c[:4] in equipe])
    A = row['End Y']-row['Start Y']/row['End X']-row['Start X']
    C = row['Start Y']-A*row['Start X']
    B = -1 
    players_danger = {}
    for player in players:
        dist = abs(A*(row[equipe+"_"+player+"_" + "x"])+B*(row[equipe+"_"+player+"_" + "y"])+ C)/np.sqrt(A**2+B**2)
        players_danger[player] = dist
    minval = min(players_danger.values())
    for k, v in players_danger.items():
        if v==minval and k!= np.nan:
            jou_plus_proche = k
            if jou_plus_proche in player_home:
                d = abs(A*(row["Home"+"_"+jou_plus_proche+"_" + "x"])+B*(row["Home"+"_"+jou_plus_proche+"_" + "y"])+ C)/np.sqrt(A**2+B**2)
            else:
                d = abs(A*(row["Away"+"_"+jou_plus_proche+"_" + "x"])+B*(row["Away"+"_"+jou_plus_proche+"_" + "y"])+ C)/np.sqrt(A**2+B**2)
    inter_model.at[i,'joueur_plus_proche'] = jou_plus_proche
    inter_model.at[i, 'dist_jou_plus_proche'] = d
       
    #inter_model.at[i, 'dist_jou_plus_proche'] = minval
    
    # Nearest player speed and position:
    
for i, row in inter_model.iterrows():
    player_home = np.unique([c.split('_')[1] for c in inter_model.columns if c[:4] in 'Home'])
    player_away = np.unique([c.split('_')[1] for c in inter_model.columns if c[:4] in 'Away'])
    if not np.isnan(row['joueur_plus_proche']):
        player = str(int(row['joueur_plus_proche']))
        if player in player_home:
            inter_model.at[i, 'vitesse_jou_proche'] = row['Home' + "_" + player + '_' + 'speed']
            inter_model.at[i, 'position_jou_proche_x'] = row['Home' + "_" + player + '_' + 'x']
            inter_model.at[i, 'position_jou_proche_y'] = row['Home' + "_" + player + '_' + 'y']
        else:
            inter_model.at[i, 'vitesse_jou_proche'] = row['Away' + "_" + player + '_' + 'speed']
            inter_model.at[i, 'position_jou_proche_x'] = row['Away' + "_" + player + '_' + 'x']
            inter_model.at[i, 'position_jou_proche_y'] = row['Away' + "_" + player + '_' + 'y']
            
    # 'From' Player's position and speed:
            
    player = row['From'][6:]
    if player != '29' and player != '30' and player != '31' and player !='32' and player != '33' and player != '34' and player != '35':
        if player in player_home:
            inter_model.at[i, 'vitesse_porteur'] = row['Home' + "_" + player + '_' + 'speed']
            inter_model.at[i, 'position_porteur_x'] = row['Home' + "_" + player + '_' + 'x']
            inter_model.at[i, 'position_porteur_y'] = row['Home' + "_" + player + '_' + 'y']
        else:
            inter_model.at[i, 'vitesse_porteur'] = row['Away' + "_" + player + '_' + 'speed']
            inter_model.at[i, 'position_porteur_x'] = row['Away' + "_" + player + '_' + 'x']
            inter_model.at[i, 'position_porteur_y'] = row['Away' + "_" + player + '_' + 'y']
    else:
        inter_model.at[i, 'vitesse_porteur'] = np.nan
        inter_model.at[i, 'position_porteur_x'] = np.nan
        inter_model.at[i, 'position_porteur_y'] = np.nan
        
    # Player's 'To' position and speed:
          
    if isinstance(row["To"], str):    
        player = row['To'][6:]
        if player in player_home:
            inter_model.at[i, 'vitesse_recepteur'] = row['Home' + "_" + player + '_' + 'speed']
            inter_model.at[i, 'position_recepteur_x'] = row['Home' + "_" + player + '_' + 'x']
            inter_model.at[i, 'position_recepteur_y'] = row['Home' + "_" + player + '_' + 'y']
        else:
            inter_model.at[i, 'vitesse_recepteur'] = row['Away' + "_" + player + '_' + 'speed']
            inter_model.at[i, 'position_recepteur_x'] = row['Away' + "_" + player + '_' + 'x']
            inter_model.at[i, 'position_recepteur_y'] = row['Away' + "_" + player + '_' + 'y']
            
    # Distance between the player with the ball and the player reciving the ball
    player_from = row['From'][6:]
    player_home = np.unique([c.split('_')[1] for c in total.columns if c[:4] in 'Home'])
    player_away = np.unique([c.split('_')[1] for c in total.columns if c[:4] in 'Away'])
    if isinstance(row["To"], str):
        player_to = row['To'][6:]
    if player_from in player_home and player_to in player_home:
        dist = np.sqrt((row['Home'+'_'+player_to+'_'+'y'] - row['Home'+'_'+player_from+'_'+'y'])**2+(row['Home'+'_'+player_to+'_'+'x'] - row['Home'+'_'+player_from+'_'+'x'])**2)
    if player_from in player_home and player_to in player_away:
        dist = np.sqrt((row['Away'+'_'+player_to+'_'+'y'] - row['Home'+'_'+player_from+'_'+'y'])**2+(row['Away'+'_'+player_to+'_'+'x'] - row['Home'+'_'+player_from+'_'+'x'])**2)
    if player_from in player_away and player_to in player_away:
        dist = np.sqrt((row['Away'+'_'+player_to+'_'+'y'] - row['Away'+'_'+player_from+'_'+'y'])**2+(row['Away'+'_'+player_to+'_'+'x'] - row['Away'+'_'+player_from+'_'+'x'])**2)
    if player_from in player_away and player_to in player_home:
        dist = np.sqrt((row['Home'+'_'+player_to+'_'+'y'] - row['Away'+'_'+player_from+'_'+'y'])**2+(row['Home'+'_'+player_to+'_'+'x'] - row['Away'+'_'+player_from+'_'+'x'])**2)
    inter_model.at[i, 'dist_port_jou'] = dist
        
    #Interception: 

    if i < inter_model.shape[0]-1:
        if inter_model.at[i, "Team"] == inter_model.at[i+1, "Team"]:
            inter_model.at[i+1, "Interception"] = 0
        else:
            inter_model.at[i+1, "Interception"] = 1

            
for column in inter_model.columns:
    if "Home" in column or "Away" in column or column == "Period_x" or column == 'Period_y' or column == "To" or column == "From" or column == 'Time [s]_y' or column == 'Time [s]_x' or column == "Attaque" or column == "Defense" or column == 'Start Frame' or column == 'End Frame' or column =='joueur_plus_proche':
        inter_model = inter_model.drop(column, axis = 1)
        

# Correlation:

import matplotlib.pyplot as plt
import seaborn as sns

cor = inter_model.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(cor, annot=True, ax=ax, cmap="coolwarm");       

#Xgboost

# We are going to make a model using the xgboost algorithm

from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

inter_model.replace([np.inf, -np.inf], np.nan, inplace=True)
inter_model['Interception'] = inter_model['Interception'].fillna(0)

features = inter_model.drop('Interception', axis=1)
features_matrix= pd.get_dummies(features)

target = inter_model.Interception

# We separate the train and test sets

X_train, X_test, y_train, y_test  = train_test_split(features_matrix, target, test_size=0.2, random_state=1234)

# As we have differents scales we are going to standardize the numeric variables.

scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)   

X_train_col = X_train.columns

X_test_col= X_test.columns
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_col)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test_col)

train = xgb.DMatrix(data=X_train_scaled, label=y_train)
test = xgb.DMatrix(data=X_test_scaled, label=y_test)

# We seach for the best hyperparameters

from sklearn.metrics import accuracy_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }

def objective(space):
    clf=xgb.XGBClassifier(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( X_train_scaled, y_train), ( X_test_scaled, y_test)]
    
    clf.fit(X_train_scaled, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)
    

    pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, pred>0.5)
    #print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)

best_hyperparams['max_depth']=best_hyperparams['max_depth'].astype('int')
params = {'booster':'gbtree', 'learning_rate': .001, 'objective': 'binary:logistic'}
params.update(best_hyperparams)
# , 'colsample_bytree':  0.8611744629391216, 'gamma':  2.5687624183220406, 'max_depth': 15, 'min_child_weight': 3.0, 'reg_alpha': 60.0, 'reg_lambda': 0.34638620207553705}

xgb1 = xgb.train(params = params, dtrain = train, num_boost_round= 4000, evals= [(train, 'train'), (test, 'test')])

# We look for the predictions:

preds = xgb1.predict(test)
xgbpreds= pd.Series(np.where(preds > 0.5, 1, 0))
print(pd.crosstab(xgbpreds, pd.Series(y_test), rownames=['Classe prédite'], colnames=['Classe réelle']))



types= ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
for f in types:
    xgb.plot_importance(xgb1, max_num_features=15, importance_type=f, title='importance: '+f);



