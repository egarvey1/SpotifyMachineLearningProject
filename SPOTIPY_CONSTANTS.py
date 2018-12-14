# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:33:13 2018

@author: PC
"""

##ALice's Info

ALICE_USER_ID='8IerXpXNRTi2c7wjvhNjLw' 
ALICE_0 = "1vHEk3Nh0Gjx73vNztvscF?si=lh2px58DQG6RC_perpcm-Q"
ALICE_1 = "5vuibiOP3vO5HuzklmvY9Y?si=0urlNxwURKyBW6iUxRRNXA "
ALICE_2 =  "00nZcOcq9UV0QwROASAOpI?si=wAnmhjpHS1GI1Vt3ZyxS8A"
ALICE_TOPSONGS = "37i9dQZF1E9XUYEJrMuH1n?si=RlllDkdTTZWEjXqe-LYqwg"
ALICE_PLAYLISTS = [ALICE_0,ALICE_1,ALICE_2,ALICE_TOPSONGS]


ALICE_CLIENT_ID = "9c378d8beda041709f63d864b29e7947"
ALICE_CLIENT_SECRET = "5175127e70ab49fea21abf87b2900081"
ALICE_DROPS = ["acousticness", "instrumentalness", "tempo"]



##Emma's Info

EMMA_USER_ID="2EUV-i_3QVeY9ousfkZNUg"
EMMA_0 = "0TDDl464s0PS9TbAjP7rqr?si=-F2jZbusQ8uFVaj9y30aBg"
EMMA_1 = "7gaKRZ6JceIn3cJPGOigNu?si=V2tDtr6VQCKAEJhWUYKe4w"
EMMA_2 =  "0TDDl464s0PS9TbAjP7rqr?si=9brHPeWhRBqBxBj_5iDTCw"
EMMA_TOPSONGS = "37i9dQZF1E9IxGjWvGDkSw?si=ULs29N5RSLGX5gkxPNZBWQ"
EMMA_PLAYLISTS = [EMMA_0, EMMA_1, EMMA_2,EMMA_TOPSONGS]


EMMA_CLIENT_ID = "58f54199a5f84104988cecd720b6ee5c"
EMMA_CLIENT_SECRET = "5b11156365b94cbdb19eec79c46eacfc" 
EMMA_DROPS = ["danceability", "valence"]


##Dictionary Keys
rating_dict = {'hate':0,'ok':1,'love':2,'top2017':2}

dict_keys = ['hate','ok','love','top2017']


##Attributes which need to be sorted
attributes_0_1float=["acousticness", "danceability", "energy",
            "instrumentalness", "liveness","speechiness", 
              "valence"]

attributes_int_range=["loudness","tempo"]

attributes_all = ["acousticness", "danceability", "energy",
            "instrumentalness","key", "liveness", "loudness", "mode", 
            "speechiness", "tempo", "time_signature", "valence"]

unused_cols = ["duration_ms", "id", "track_href", "type", "uri",
               "rating", "analysis_url"]


