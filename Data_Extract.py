# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:14:06 2018

@author: PC
"""
import pandas as pd 
import numpy as np
import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials 
from SPOTIPY_CONSTANTS import *


class data_loader():
    def __init__(self, user_ID, client_ID, secret_ID, playlist_IDs, num_splits):
        self.user_ID = user_ID
        self.client_ID = client_ID
        self.secret_ID = secret_ID
        self.playlist_IDs = playlist_IDs
        self.num_splits = num_splits
        self.playlist_dataset=None
        


        
        self.credentials_manager = SpotifyClientCredentials(
                                                    client_id=self.client_ID,
                                                client_secret=self.secret_ID)
        
        self.sp = spotipy.Spotify(client_credentials_manager=
                                          self.credentials_manager) 
        self.sp.trace=False
        
        self.playlist_dict = {key: None for key in dict_keys}   
        
        
        self.load_playlists()
        
        self.sort_playlists()
        self.compile_data()
#        self.remove_duplicates()
        
    def load_playlists(self):
        for playlist_id, key in zip(self.playlist_IDs, self.playlist_dict):
            playlist = self.sp.user_playlist(self.user_ID, playlist_id,
                                             fields="tracks, next")
            
            songs = playlist["tracks"]["items"] 
            ids=[]
            features_list= []
            for i in range(len(songs)):
                ids.append(songs[i]['track']['id'])
                
            features = self.sp.audio_features(ids)
            features_list.extend(features)

            tracks = playlist['tracks']['items']            
            playlist = playlist['tracks']
            
            while playlist['next']:
                playlist = self.sp.next(playlist)
                tracks.extend(playlist['items'])  
                
                songs=playlist['items']
                
                ids=[]
                for i in range(len(songs)):
                    ids.append(songs[i]["track"]["id"])
                    
                
                features = self.sp.audio_features(ids)
                features_list.extend(features)               

            self.playlist_dict[key] = pd.DataFrame(features_list)
            
    def remove_duplicates(self):
        print(len(self.playlist_dataset))
        self.playlist_dataset.drop_duplicates(subset='id')
        print(len(self.playlist_dataset))
            
            
    def sort_playlists(self):
        
        for key in self.playlist_dict:
            playlist_df = self.playlist_dict[key]
            num_songs = len(playlist_df)
            playlist_df["rating"] = np.full((num_songs, 1), rating_dict[key])
            self.map_values(key)
                
    def map_values(self, key):
        
        playlist_df=self.playlist_dict[key]
        
        for attribute in attributes_all:
            column = playlist_df[attribute]
            
            if attribute in attributes_0_1float:
                index_vals = list(range(2,self.num_splits+2))
                range_vals = np.linspace(1/self.num_splits, 1, self.num_splits)

            if attribute in attributes_int_range:
                index_vals = list(range(230, 230+self.num_splits))
                
                if attribute =="loudness":
                    range_vals = np.linspace(-60, 0, self.num_splits)
                    
                    
                if attribute == "tempo":
                    range_vals = np.linspace(0, 230, self.num_splits)
                    
                    
                    
            for result,val_range in zip(index_vals, range_vals):
                    indices = np.where(column <= val_range) 
                    np.put(column, indices, result) 
                    
            
            
        
    def compile_data(self):
        
        frames = []
        for key in self.playlist_dict:
            frames.append(self.playlist_dict[key])
        self.playlist_dataset = pd.concat(frames)
            
            
            
        
        
        
        
        
        
        
if __name__ == '__main__':
    
    alice= data_loader(ALICE_USER_ID, ALICE_CLIENT_ID, ALICE_CLIENT_SECRET, 
                       ALICE_PLAYLISTS, 6)
    
    alice_data = alice.playlist_dataset
#    
    emma = data_loader(EMMA_USER_ID, EMMA_CLIENT_ID, EMMA_CLIENT_SECRET,
                       EMMA_PLAYLISTS, 8)
    
    emma_data = emma.playlist_dataset
        
        
        
        
        
        
        
        
        
        
        