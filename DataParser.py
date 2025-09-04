# Player Game Data Parser
import pandas as pd
import re

capitalized_words = r"((?:[A-Z][a-z']+ ?)+)" #regex to get capitalized words in sentence
hitter_by_game_df = pd.read_csv('hittersByGame(player_offense_data).csv')
pitcher_by_game_df = pd.read_csv('pitchersByGame(pitcher_data).csv')
baserunning_by_game_df = pd.read_csv('baserunningNotes(player_offense_data).csv')
fielding_by_game_df = pd.read_csv('fieldingNotes(player_defensive_data).csv')
warp_hitter_df = pd.read_csv('bp_hitters_2021.csv')
warp_pitcher_df = pd.read_csv('bp_pitchers_2021.csv')
oaa_hitter_df = pd.read_csv('outs_above_average.csv')
fielding_df = pd.read_csv('fieldingNotes(player_defensive_data).csv')
baserunning_df = pd.read_csv('baserunningNotes(player_offense_data).csv')
war_df = pd.read_csv('FanGraphs Leaderboard.csv')

#gets rid of unnecessary columns in hitter csv
def clean_sorted_hitter():
	hitter_by_game_df.drop(['H-AB', 'AB', 'H', '#P', 'Game', 'Team', 'Hitter Id'], axis = 1)
	sorted_hitter_df = hitter_by_game_df.sort_values(by='Hitters')
	return sorted_hitter_df

#gets rids of unnecessary columns in pitcher csv
def clean_sorted_pitcher():
	pitcher_by_game_df.drop(['R', 'ER', 'PC', 'Game', 'Team', 'Extra', 'Pitcher Id'], axis = 1)
	sorted_pitcher_df = pitcher_by_game_df.sort_values(by='Pitchers')
	return sorted_pitcher_df

def clean_warp_hitter():
	warp_hitter_df.drop(['bpid', 'mlbid', 'Age', 'DRC+', '+/-', 'PA', 'R', 'RBI','ISO', 'K%', 'BB%','Whiff%'], axis = 1)
	sorted_warphitter_df = warp_hitter_df.sort_values(by='WARP')
	return sorted_warphitter_df

def clean_warp_pitcher():
	warp_pitcher_df.drop(['bpid', 'mlbid', 'DRA-', 'DRA', 'DRA SD', 'cFIP', 'GS', 'W','L', 'ERA', 'RA9','Whiff%'], axis = 1)
	sorted_warppitcher_df = warp_pitcher_df.sort_values(by='WARP')
	return sorted_warppitcher_df

def clean_war():
	war_df.drop(['playerid','Team','Pos'], axis = 1)
	sorted_war_df = war_df.sort_values(by='Total WAR')
	return sorted_war_df

#gets rids of unnececessary baserunning columns
def clean_sorted_baserunning():
	del baserunning_by_game_df['Game']
	sorted_baserunning_df = baserunning_by_game_df.sort_values(by='Stat')
	scores = []
	playerscurrent = []
	playerstotal = []
	runsadded = []	
	for key,value in sorted_baserunning_df.iteritems():
		statlines = value.split(',')
		if statlines[0] == 'SB':
			for x in int(statlines[1][:1]):
				playerscurrent = re.findall(capitalized_words, statlines[x+2])
			for x in range(0,len(playerscurrent)):
				for y in range(0,len(playerstotal)):
					if playerscurrent[x] == playerstotal[y]:
						runsadded[x]+=1
					else:
						playerstotal.append(playerscurrent[x])
						runsadded.append(1)
		elif statlines[0] == 'CS' | statlines[0] == 'Picked Off':
			for x in int(statlines[1][:1]):
				playerscurrent = re.findall(capitalized_words, statlines[x+2])
			for x in range(0,len(playerscurrent)):
				for y in range(0,len(playerstotal)):
					if playerscurrent[x] == playerstotal[y]:
						runsadded[x]-=1
					else:
						playerstotal.append(playerscurrent[x])
						
	for x in range(0, len(runsadded)):
		scores[x] = (runsadded[x] / 3)

	baserunningvalues_hash_table = HashTable(len(scores))
	for x in range(0,len(scores)):
		baserunningvalues_hash_table.set_val(playerstotal[x], scores[x])
	return baserunningvalues_hash_table


def clean_defensive_players():
	oaa_hitter_df.drop(['player_id','display_team_name','year','primary_pos_formatted'],axis = 1)
	fielding_df.drop(['Game','Team'])
	sorted_fielding_df = fielding_df.sort_values(by='Stat')

	scores = []
	playerscurrent = []
	playerstotal = []
	outssaved = []

	#TODO, assigning values for various defensive plays to the players that participated in them, keep track of them via secondary array (hashmap w/ key and value?)
	for key,value in sorted_fielding_df.iteritems():
		statlines = value.split(',')
		if statlines[0] == 'DP':
			for x in int(statlines[1][:1]):
				playerscurrent = re.findall(capitalized_words, statlines[x+2])
			for x in range(0,len(playerscurrent)):
				for y in range(0,len(playerstotal)):
					if playerscurrent[x] == playerstotal[y]:
						outssaved[x]+=1
					else:
						playerstotal.append(playerscurrent[x])
						outssaved.append(1)
		elif statlines[0] == 'Assists':
			for x in int(statlines[1][:1]):
				playerscurrent = re.findall(capitalized_words, statlines[x+2])
			for x in range(0,len(playerscurrent)):
				for y in range(0,len(playerstotal)):
					if playerscurrent[x] == playerstotal[y]:
						outssaved[x]+=.5
					else:
						playerstotal.append(playerscurrent[x])
						outssaved.append(.5)
		elif statlines[0] == 'E':
			for x in int(statlines[1][:1]):
				playerscurrent = re.findall(capitalized_words, statlines[x+2])
			for x in range(0,len(playerscurrent)):
				for y in range(0,len(playerstotal)):
					if playerscurrent[x] == playerstotal[y]:
						outssaved[x]-=1
					else:
						playerstotal.append(playerscurrent[x])
						outssaved.append(-1) 
	for x in range(0, len(outssaved)):
		scores[x] = (outssaved[x] / 3)

	defensivevalues_hash_table = HashTable(len(scores))
	for x in range(0,len(scores)):
		defensivevalues_hash_table.set_val(playerstotal[x], scores[x])
	return defensivevalues_hash_table

class HashTable:

	# Create empty bucket list of given size
	def __init__(self, size):
		self.size = size
		self.hash_table = self.create_buckets()

	def create_buckets(self):
		return [[] for _ in range(self.size)]

	# Insert values into hash map
	def set_val(self, key, val):
		
		# Get the index from the key
		# using hash function
		hashed_key = hash(key) % self.size
		
		# Get the bucket corresponding to index
		bucket = self.hash_table[hashed_key]

		found_key = False
		for index, record in enumerate(bucket):
			record_key, record_val = record
			
			# check if the bucket has same key as
			# the key to be inserted
			if record_key == key:
				found_key = True
				break

		# If the bucket has same key as the key to be inserted,
		# Update the key value
		# Otherwise append the new key-value pair to the bucket
		if found_key:
			bucket[index] = (key, val)
		else:
			bucket.append((key, val))

	# Return searched value with specific key
	def get_val(self, key):
		
		# Get the index from the key using
		# hash function
		hashed_key = hash(key) % self.size
		
		# Get the bucket corresponding to index
		bucket = self.hash_table[hashed_key]

		found_key = False
		for index, record in enumerate(bucket):
			record_key, record_val = record
			
			# check if the bucket has same key as
			# the key being searched
			if record_key == key:
				found_key = True
				break

		# If the bucket has same key as the key being searched,
		# Return the value found
		# Otherwise indicate there was no record found
		if found_key:
			return record_val
		else:
			return "No record found"

	# Remove a value with specific key
	def delete_val(self, key):
		
		# Get the index from the key using
		# hash function
		hashed_key = hash(key) % self.size
		
		# Get the bucket corresponding to index
		bucket = self.hash_table[hashed_key]

		found_key = False
		for index, record in enumerate(bucket):
			record_key, record_val = record
			
			# check if the bucket has same key as
			# the key to be deleted
			if record_key == key:
				found_key = True
				break
		if found_key:
			bucket.pop(index)
		

	# To print the items of hash map
	def __str__(self):
		return "".join(str(item) for item in self.hash_table)
