import pandas as pd
from deprecated.DataParser import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error,r2_score, mean_squared_error
import plotly.express as px

#first attempt approach with hitter data/stats, then depending on efficacy tweak & add defensive stats
	# NOTE -> hitting stats include (walks, pitches taken, average, obp+slug)
	# DOUBLE NOTE-> defensive stats will have to only include fielding (outcomes of fielding are (double play, assist, error, passed ball *catcher specific might disregard for now*)
	# TRIPLE NOTE-> add baserunning as a variable as well(outcomes of baserunning are stealing, caught stealing, picked off)
	# after hitter stats attempt, run again w/ pitcher data
	# as we don't have framing data currently it might be better to include catchers as just typical positional players
	# QUADRUPLE NOTE-> after further data exploration, this is boiling down into total runs scored + total runs saved into a predictor for player performance -> WARP + defensive stat created, compare to WAR
	# QUINTUPLE NOTE-> For pitchers, it boils down to total runs saved through pitching really
	# pulling data for models from the cleaned data from the dataparser program

def performancelinear():
	x_warp_train,x_warp_test,y_warp_train,y_warp_test,x_war_train,x_war_test,y_war_train,y_war_test,a_warp_train,a_warp_test,b_warp_train,b_warp_test,a_war_train, a_war_test, b_war_train, b_war_test = data_preparation()
	regressor = LinearRegression()
	regressor.fit(x_warp_train, y_warp_train)
	y_warp_pred = regressor.predict(x_warp_test)
	print("R2Linear - hitter WARP correlation: " + r2_score(y_warp_test,y_warp_pred) + ", RMSELinear - hitter WARP correlation: " + mean_squared_error(y_warp_test,y_warp_pred))
	regressor.fit(x_war_train, y_war_train)
	y_war_pred = regressor.predict(x_war_test)
	print("R2Linear - hitter WAR correlation: " + r2_score(y_war_test,y_war_pred) + ", RMSELinear - hitter WAR correlation: " + mean_squared_error(y_war_test,y_war_pred))
	regressor.fit(a_warp_train, b_warp_train)
	b_warp_pred = regressor.predict(a_warp_test)
	print("R2Linear - pitcher WARP correlation: " + r2_score(b_warp_test,b_warp_pred) + ", RMSELinear - pitcher WARP correlation: " + mean_squared_error(b_warp_test,b_warp_pred))
	regressor.fit(a_war_train, b_war_train)
	b_war_pred = regressor.predict(a_war_test)
	print("R2Linear - pitcher WAR correlation: " + r2_score(b_war_test,b_war_pred) + ", RMSELinear - pitcher WAR correlation: " + mean_squared_error(b_war_test,b_war_pred))
	figure  = px.scatter(y_warp_pred,y_warp_test, x = 'Predicted Performance Hitter(Linear)', y = 'Actual Performance Hitter(WARP)', hover_name = y_warp_test-y_warp_pred, title = 'Actual Performance Hitter vs. Pred. Performance')
	figure.show()
	figure  = px.scatter(y_war_pred,y_war_test, x = 'Predicted Performance Hitter(Linear)', y = 'Actual Performance Hitter(WAR)', hover_name = y_war_test-y_war_pred, title = 'Actual Performance Hitter vs. Pred. Performance')
	figure.show()
	figure  = px.scatter(b_warp_pred,b_warp_test, x = 'Predicted Performance Pitcher(Linear)', y = 'Actual Performance Pitcher(WARP)', hover_name = b_warp_test-b_warp_pred, title = 'Actual Performance Pitcher vs. Pred. Performance')
	figure.show()
	figure  = px.scatter(b_war_pred,b_war_test, x = 'Predicted Performance Pitcher(Linear)', y = 'Actual Performance Pitcher(WAR)', hover_name = b_war_test-b_war_pred, title = 'Actual Performance Pitcher vs. Pred. Performance')
	figure.show()    

def performancelasso():
	x_warp_train,x_warp_test,y_warp_train,y_warp_test,x_war_train,x_war_test,y_war_train,y_war_test,a_warp_train,a_warp_test,b_warp_train,b_warp_test,a_war_train, a_war_test, b_war_train, b_war_test = data_preparation()
	regressor = Lasso()
	regressor.fit(x_warp_train, y_warp_train)
	y_warp_pred = regressor.predict(x_warp_test)
	print("R2Lasso - hitter WARP correlation: " + r2_score(y_warp_test,y_warp_pred) + ", RMSELasso - hitter WARP correlation: " + mean_squared_error(y_warp_test,y_warp_pred))
	regressor.fit(x_war_train, y_war_train)
	y_war_pred = regressor.predict(x_war_test)
	print("R2Lasso - hitter WAR correlation: " + r2_score(y_war_test,y_war_pred) + ", RMSELasso - hitter WAR correlation: " + mean_squared_error(y_war_test,y_war_pred))
	regressor.fit(a_warp_train, b_warp_train)
	b_warp_pred = regressor.predict(a_warp_test)
	print("R2Lasso - pitcher WARP correlation: " + r2_score(b_warp_test,b_warp_pred) + ", RMSELasso - pitcher WARP correlation: " + mean_squared_error(b_warp_test,b_warp_pred))
	regressor.fit(a_war_train, b_war_train)
	b_war_pred = regressor.predict(a_war_test)
	print("R2Lasso - pitcher WAR correlation: " + r2_score(b_war_test,b_war_pred) + ", RMSELasso - pitcher WAR correlation: " + mean_squared_error(b_war_test,b_war_pred))
	figure  = px.scatter(y_warp_pred,y_warp_test, x = 'Predicted Performance Hitter(Lasso)', y = 'Actual Performance Hitter(WARP)', hover_name = y_warp_test-y_warp_pred, title = 'Actual Performance Hitter vs. Pred. Performance')
	figure.show()
	figure  = px.scatter(y_war_pred,y_war_test, x = 'Predicted Performance Hitter(Lasso)', y = 'Actual Performance Hitter(WAR)', hover_name = y_war_test-y_war_pred, title = 'Actual Performance Hitter vs. Pred. Performance')
	figure.show()
	figure  = px.scatter(b_warp_pred,b_warp_test, x = 'Predicted Performance Pitcher(Lasso)', y = 'Actual Performance Pitcher(WARP)', hover_name = b_warp_test-b_warp_pred, title = 'Actual Performance Pitcher vs. Pred. Performance')
	figure.show()
	figure  = px.scatter(b_war_pred,b_war_test, x = 'Predicted Performance Pitcher(Lasso)', y = 'Actual Performance Pitcher(WAR)', hover_name = b_war_test-b_war_pred, title = 'Actual Performance Pitcher vs. Pred. Performance')
	figure.show()

def performanceelasticnet():
	x_warp_train,x_warp_test,y_warp_train,y_warp_test,x_war_train,x_war_test,y_war_train,y_war_test,a_warp_train,a_warp_test,b_warp_train,b_warp_test,a_war_train, a_war_test, b_war_train, b_war_test = data_preparation()
	regressor = ElasticNet()
	regressor.fit(x_warp_train,y_warp_train)
	y_warp_pred = regressor.predict(x_warp_test)
	print("R2Elastic - hitter WARP correlation: " + r2_score(y_warp_test,y_warp_pred) + ", RMSEElastic - hitter WARP correlation: " + mean_squared_error(y_warp_test,y_warp_pred))
	regressor.fit(x_war_train,y_war_train)
	y_war_pred = regressor.predict(x_war_test)
	print("R2Elastic - hitter WAR correlation: " + r2_score(y_war_test,y_war_pred) + ", RMSEElastic - hitter WAR correlation: " + mean_squared_error(y_war_test,y_war_pred))
	regressor.fit(a_warp_train, b_warp_train)
	b_warp_pred = regressor.predict(a_warp_test)
	print("R2Elastic - pitcher WARP correlation: " + r2_score(b_warp_test,b_warp_pred) + ", RMSEElastic - pitcher WARP correlation: " + mean_squared_error(b_warp_test,b_warp_pred))
	regressor.fit(a_war_train, b_war_train)
	b_war_pred = regressor.predict(a_war_test)
	print("R2Elastic - pitcher WAR correlation: " + r2_score(b_war_test,b_war_pred) + ", RMSEElastic - pitcher WAR correlation: " + mean_squared_error(b_war_test,b_war_pred))
	figure  = px.scatter(y_warp_pred,y_warp_test, x = 'Predicted Performance Hitter(Elastic Net)', y = 'Actual Performance Hitter(WARP)', hover_name = y_warp_test-y_warp_pred, title = 'Actual Performance Hitter vs. Pred. Performance')
	figure.show()
	figure  = px.scatter(y_war_pred,y_war_test, x = 'Predicted Performance Hitter(Elastic Net)', y = 'Actual Performance Hitter(WAR)', hover_name = y_war_test-y_war_pred, title = 'Actual Performance Hitter vs. Pred. Performance')
	figure.show()
	figure  = px.scatter(b_warp_pred,b_warp_test, x = 'Predicted Performance Pitcher(Elastic Net)', y = 'Actual Performance Pitcher(WARP)', hover_name = y_warp_test-y_warp_pred, title = 'Actual Performance Pitcher vs. Pred. Performance')
	figure.show()
	figure  = px.scatter(b_war_pred,b_war_test, x = 'Predicted Performance Pitcher(Elastic Net)', y = 'Actual Performance Pitcher(WAR)', hover_name = y_war_test-y_war_pred, title = 'Actual Performance Pitcher vs. Pred. Performance')
	figure.show()

def performanceknnvisualization():
	x_warp_train,x_warp_test,y_warp_train,y_warp_test,x_war_train,x_war_test,y_war_train,y_war_test,a_warp_train,a_warp_test,b_warp_train,b_warp_test,a_war_train, a_war_test, b_war_train, b_war_test = data_preparation()
	model_warp_hitter = KNeighborsClassifier(n_neighbors = 3)
	model_warp_hitter.fit(x_warp_train,y_warp_train)
	model_warp_pitcher = KNeighborsClassifier(n_neighbors = 3)
	model_warp_pitcher.fit(a_warp_train,b_warp_train)
	model_war_hitter = KNeighborsClassifier(n_neighbors = 3)
	model_war_hitter.fit(x_war_train,y_war_train)
	model_war_pitcher = KNeighborsClassifier(n_neighbors = 3)
	model_war_pitcher.fit(a_war_train,b_war_train)
	print("Accuracy Score - KNN WARP hitters: " + model_warp_hitter.score(x_warp_test,y_warp_test))
	print("Accuracy Score - KNN WARP pitchers: " + model_warp_pitcher.score(a_warp_test,b_warp_test))
	print("Accuracy Score - KNN WAR hitters: " + model_war_hitter.score(x_war_test,y_war_test))
	print("Accuracy Score - KNN WAR pitchers: " + model_war_pitcher.score(a_war_test,b_war_test))
   
def data_preparation():
	hitter_data = clean_sorted_hitter()
	hitter_pred_data = clean_warp_hitter()
	pitcher_data = clean_sorted_pitcher()
	pitcher_pred_data = clean_warp_pitcher()
	#defensive_values = clean_defensive_players() 
	#baserunning_values = clean_sorted_baserunning() 
	war_values = clean_war()

	x_warp = []
	y_warp = []
	x_war = []
	y_war = []
	#multiple linear regression where x is composed of multiple variables

	## Possible solution that accounts for the disproportionate amount of values in the hitter_data csv compared to the hitter_pred_Data csv, UNTESTED
	for index,row in hitter_data.iterrows():
		name = row["Hitters"]
		if hitter_pred_data['Name'].values[0]==name:
			x_warp += hitter_data[name,['K','BB','AVG','OBP','SLG']]
			x_war += hitter_data[name,['K','BB','AVG','OBP','SLG']]
			# y is pulled from a separate database that I pulled to actually get the x variables to predict a "performance number" rather than a correlation between two statistics 
			y_warp += hitter_pred_data['WARP']
			y_war+=war_values['Total War']

	# previous lines 123-125 for Ms. Yao
	for index,row in hitter_pred_data.iterrows():
		name = row['Name']
		if hitter_data['Hitters'].values[0]==name:
			x_warp += hitter_data[name,['K','BB','AVG','OBP','SLG']]
			x_war += hitter_data[name,['K','BB','AVG','OBP','SLG']]
			# y is pulled from a separate database that I pulled to actually get the x variables to predict a "performance number" rather than a correlation between two statistics 
			y_warp += hitter_pred_data['WARP']
			y_war+=war_values['Total War']
		#if defensive_values.get_val(name) != "No record found":
		#	x_war+= defensive_values.get_val(name)
		#if baserunning_values.get_val(name) != "No record found":
	#		x_war+= baserunning_values.get_val(name)

	# add additional factors based off of rows in the relevant cavs, (player.csv for players, pitcher.csv for pitchers)
	x_warp_train, x_warp_test, y_warp_train, y_warp_test = train_test_split(x_warp,y_warp, test_size=.25,train_size = .75,random_state=1)
	x_war_train, x_war_test, y_war_train, y_war_test = train_test_split(x_war, y_war, test_size = .25,train_size = .75, random_state=1)
	a_warp = []
	b_warp = []
	a_war=[]
	b_war = []
	for row in pitcher_pred_data.iterrows():
		name = row['Name']
		if pitcher_data['Pitchers'].values[0]==name:
			a_warp += pitcher_data[name,['IP', 'BB','K','HR','ERA']]
			# y is pulled from a separate database that I pulled to actually get the x variables to predict a "performance number" rather than a correlation between two statistics 
			b_warp+= pitcher_pred_data['WARP']
			b_war+=war_values['Primary WAR']
	a_warp_train, a_warp_test, b_warp_train, b_warp_test = train_test_split(a_warp,b_warp, test_size=.25,train_size = .75,random_state=1)
	a_war_train, a_war_test, b_war_train, b_war_test = train_test_split(a_war,b_war, test_size=.25,train_size = .75,random_state=1)
	

	return (x_warp_train,x_warp_test,y_warp_train,y_warp_test,x_war_train,x_war_test,y_war_train,y_war_test,a_warp_train,a_warp_test,b_warp_train,b_warp_test,a_war_train,a_war_test, b_war_train, b_war_test)


def main():
	performancelinear()
	performancelasso()
	performanceelasticnet()
	performanceknnvisualization()

if __name__ == "__main__":
	main()

