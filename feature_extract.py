import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from nltk.stem import PorterStemmer
from sklearn.datasets import dump_svmlight_file
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import cross_val_score

#getting familiar with the data
print()
print("train data first ten entreis")
train_data = pd.read_csv("train.csv")
#prints first ten of training data. can see theres an id, keyword, location, text, and target
print(train_data.head(10)) 
print()

test_data = pd.read_csv("test.csv")
#print first ten of test data, can see that there is all the same but no target
print("test data first ten entreis") 
print(test_data.head(10))
print()

#print the first ten of what our sample submission file should look like. should be the id and predicted value (target)
sample_data = pd.read_csv("sample_submission.csv")
print("sample submission first ten entries") 
print(sample_data.head(10))
print()

#looking to see how many null entries there are in both data sets
print("training data null entries: ")
print(train_data.isnull().sum())
print()

print("test data null entries")
print(test_data.isnull().sum())
print()

#shows the proportion of empty values between location and keyword for train
train_data.isna().sum().plot(kind="pie")
plt.title("Amount of Empty Values in Training Data")
plt.show()

#same but for test data
test_data.isna().sum().plot(kind="pie")
plt.title("Amount of Empty Values in Test Data")
plt.show()

#getting rid of the keyword and location columns as they are not necessary
train_data = train_data.drop(["keyword","location"], axis = 1)
test_data = test_data.drop(["keyword", "location"],axis=1)

#first half of disaster words
disaster_words = ["ablaze","arson","arsonist","blaze","blazing","buildings burning","buildings on fire","burned","burning","burning buildings","bush fires","engulfed","fire","flames","smoke"
					,"accident","attack","attacked","battle","catastrophe", "catastrophic","curfew","damage","danger","debris","desolate","desolation","devastated","devastation","disaster","displaced",
					"emergency","emergency plan","epicentre","evacuate","evacuated","evacuation","eyewitness","famine","fear","harm","hazard","hazardous","hellfire","hijack","hijacker","hijacking",
					"hostage","hostages","inundated","inundation","massacre","mayhem","meltdown","pandemonium","panic","panicking","razed","refugees","riot","rioting","ruin","screamed","screaming","screams",
					"sunk","survive","survived","survivors","terrorism","terrorist","threat","tragedy","trapped","traumatised","trouble","upheaval","weapon","weapons","whirlwind","aftershock","avalanche",
					"blizzard","cyclone","deluge","deluged","drought","dust storm","earthquake","flood","flooding","floods","forest fire","forest fires","hail","hailstorm","heat wave","hurricane",
					"landslide","lava","lightning","mudslide","natural disaster","rainstorm","sandstorm","storm","seismic","sinkhole","sinking","snowstorm","snow","thunder","thunderstorm","tornado",]

#making a list of just the tweets to loop through
tweets = train_data["text"]
#making just a list of the targets 
targets = train_data["target"]

#dictionary to to hold the term as a key and the prportion it appears in a disaster tweet as the value
term_prop_dict = {}
for disaster_word in range(len(disaster_words)):
	#counts if the given keyword was in a tweet where the target was 1, meaning it was about a disaster, resets for every word
	keyword_disaster_counter = 0
	#counts every time the keyword was in a tweet, regardless of it being about a disaster or not, resets every tweet
	keyword_counter = 0
	for tweet in range(len(tweets)):
		if disaster_words[disaster_word] in tweets[tweet]:
			keyword_counter += 1
			if(targets[tweet] == 1):
				#if it is also in a tweet where the targer is 1 (is about a disaster) then increment keyword_disaster_counter by 1
				keyword_disaster_counter += 1
	term_prop_dict[disaster_words[disaster_word]] = keyword_disaster_counter / keyword_counter

#barplot of the first half of disaster keywords with their proportion of actually being for a disaster
keys = term_prop_dict.keys()
values = term_prop_dict.values()

plt.bar(keys, values)
plt.xticks(rotation=90)
plt.xlabel("First Half of Disaster Terms")
plt.ylabel("Proportion Occuring in Disaster Tweets")
plt.title("Percentage a disaster keywords actually indicates a Disaster Tweet")
plt.show()

#second half of disaster words
disaster_words_p2 = ["tsunami","twister","typhoon","violent storm","volcano","wild fires","wildfire","windstorm","airplane accident","oil spill","ambulance","army","emergency services",
					"fire truck","first responders","military","police","rescue","rescued","rescuers","siren","sirens","war zone","annihilated","annihilation","bridge collapse","collapse",
					"collapsed","collide","collided","collision","crash","crashed","crush","crushed","demolished","demolish","demolition","derail","derailed","derailment","destroy","destroyed",
					"destruction","flattened","loud bang","obliterate","obliterated","obliteration","structural failure","wreck","wreckage","wrecked", "apocalypse","armageddon", "bioterror",
					"bioterrorism","blight","chemical emergency","electrocute","electrocuted","nuclear reactor","outbreak","quarantine","quarantined","radiation","nuclear disaster", 
					"bleeding","blood","bloody","cliff","injured","injuries","injury","stretcher","trauma","wounded","wounds","blew up","blown up","bomb","bombed","bombing","detonate",
					"detonation","explode","exploded","explosion","rubble","suicide bomb","suicide bomber","suicide bombing","body bag","body bagging","body bags","casualties","casualty",
					"dead","death","deaths","drown","drowned","drowning","fatal","fatalities","fatality","mass murder","mass murderer"]

term_prop_dict = {}
for disaster_word in range(len(disaster_words_p2)):
	#counts if the given keyword was in a tweet where the target was 1, meaning it was about a disaster, resets for every word
	keyword_disaster_counter = 0
	#counts every time the keyword was in a tweet, regardless of it being about a disaster or not, resets every tweet
	keyword_counter = 0
	for tweet in range(len(tweets)):
		if disaster_words_p2[disaster_word] in tweets[tweet]:
			keyword_counter += 1
			if(targets[tweet] == 1):
				#if it is also in a tweet where the targer is 1 (is about a disaster) then increment keyword_disaster_counter by 1
				keyword_disaster_counter += 1
	term_prop_dict[disaster_words_p2[disaster_word]] = keyword_disaster_counter / keyword_counter

keys = term_prop_dict.keys()
values = term_prop_dict.values()

#bar plot for second half of keywords
plt.bar(keys, values)
plt.xticks(rotation=90)
plt.xlabel("Disaster Terms Second Half")
plt.ylabel("Proportion Occuring in Disaster Tweets")
plt.title("Percentage a disaster keywords actually indicates a Disaster Tweet")
plt.show()

#looking at the proportion of disaster vs non disaster tweets
actual_disaster = len(train_data[train_data["target"] == 1])
percent_disaster = actual_disaster/train_data.shape[0]
not_disaster = 1 - percent_disaster

print("the percent of tweets in training data that are actually about a disaster is: " + str(percent_disaster))
print("the percent of tweets in training data that are not actually about a disaster is: " + str(not_disaster))
print()

#wordcloud for disaster tweets
disaster_tweets = train_data[train_data['target'] ==1 ]['text']
wordcloud_disaster = WordCloud( background_color='black',
                        width=500,
                        height=500).generate(" ".join(disaster_tweets))
plt.figure()
plt.imshow(wordcloud_disaster)
plt.title("Disaster Tweets Word Cloud")
plt.axis("off")
plt.show()

#wordcloud for non disaster tweets
non_disaster_tweets = train_data[train_data['target'] !=1 ]['text']
wordcloud_non_disaster = WordCloud( background_color='black',
                        width=500,
                        height=500).generate(" ".join(non_disaster_tweets))
plt.figure()
plt.imshow(wordcloud_non_disaster)
plt.title("Non-Disaster Tweets Word Cloud")
plt.axis("off")
plt.show()

##########################
#preprocessing
stop_words = nltk.corpus.stopwords.words('english')
additional_stopwords = ['.',',',' ','[',']','`',"'","@",":",';','(',')','!','"',"``","''",">","<","?","*","--","|","1","2","3","4","5","6","7","8","9","0","#"]
stop_words.extend(additional_stopwords)
ps = PorterStemmer()

#preprocessing training data
#making all the tweets go to lower case, removing links, removing puncuation. Tokenizing, cleaning for stopwords, and then stemming
for tweet in range(len(tweets)):
	tweets[tweet] = tweets[tweet].lower()
	tweets[tweet] = re.sub("https?://\S+|www\.\S+", '', tweets[tweet])
	tweets[tweet] = re.sub('<.*?>+', '', tweets[tweet])
	tweets[tweet] = re.sub('[%s]' % re.escape(string.punctuation), '', tweets[tweet])
	tokens = word_tokenize(tweets[tweet])
	clean = [word for word in tokens if word not in stop_words]
	tweets[tweet] = [ps.stem(word) for word in clean]
	tweets[tweet] = ' '.join(tweets[tweet])
print(train_data.head())
print()

#preprocessing test data
#making all the tweets go to lower case, removing links, removing puncuation. Tokenizing, cleaning for stopwords, and then stemming
test_tweets = test_data["text"]
#preprocessing test data
for tweet in range(len(test_tweets)):
	test_tweets[tweet] = test_tweets[tweet].lower()
	test_tweets[tweet] = re.sub("https?://\S+|www\.\S+", '', test_tweets[tweet])
	test_tweets[tweet] = re.sub('<.*?>+', '', test_tweets[tweet])
	test_tweets[tweet] = re.sub('[%s]' % re.escape(string.punctuation), '', test_tweets[tweet])
	tokens = word_tokenize(test_tweets[tweet])
	clean = [word for word in tokens if word not in stop_words]
	test_tweets[tweet] = [ps.stem(word) for word in clean]
	test_tweets[tweet] = ' '.join(test_tweets[tweet])
print(test_data.head())
print()

#making vectors and files for TF, IDF, TFIDF, Boolean
#term frequency for each
tf = CountVectorizer()
train_vec = tf.fit_transform(train_data["text"])
test_vec = tf.fit_transform(test_data["text"])
dump_svmlight_file(train_vec,train_data['id'],"train.tf")

#idf for each
idf = TfidfVectorizer(use_idf = True, binary = False, norm = False, ngram_range=(2,2))
train_idf = idf.fit_transform(train_data["text"])
test_idf = idf.fit_transform(test_data["text"])
dump_svmlight_file(train_idf,train_data['id'],"train.idf")

#tfidf for each
tfidf = TfidfVectorizer(min_df = 2,max_df = 0.5,ngram_range = (1,2))
train_tfidf = tfidf.fit_transform(train_data['text'])
test_tfidf = tfidf.transform(test_data['text'])
dump_svmlight_file(train_tfidf,train_data['id'],"train.tfidf")

#boolean for each
binarizer = TfidfVectorizer(use_idf = False, binary = True, norm = False, ngram_range=(2,2))
train_boolean = binarizer.fit_transform(train_data["text"])
test_boolean = binarizer.fit_transform(test_data["text"])
dump_svmlight_file(train_boolean,train_data["id"],"train.boolean")

#scoring for the multinomial naive bayes
print("Multinomial Naive Bayes scores for training data")
clf = MultinomialNB()
scores = model_selection.cross_val_score(clf,train_vec,train_data['target'],cv = 5,scoring = 'f1_macro')
print("Accuracy TF: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))

scores = model_selection.cross_val_score(clf,train_idf,train_data['target'],cv = 5,scoring = 'f1_macro')
print("Accuracy IDF: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))

scores = model_selection.cross_val_score(clf,train_tfidf,train_data['target'],cv = 5,scoring = 'f1_macro')
print("Accuracy TFIDF: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))

scores = model_selection.cross_val_score(clf,train_boolean,train_data['target'],cv = 5,scoring = 'f1_macro')
print("Accuracy Boolean: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))
print()

#logistic regression scoring
print("Logistic Regression scores for training data")
lgr = LogisticRegression()
scores = model_selection.cross_val_score(lgr,train_vec,train_data['target'],cv = 5,scoring = 'f1_macro')
print("Accuracy TF: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))

scores = model_selection.cross_val_score(lgr,train_idf,train_data['target'],cv = 5,scoring = 'f1_macro')
print("Accuracy IDF: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))

scores = model_selection.cross_val_score(lgr,train_tfidf,train_data['target'],cv = 5,scoring = 'f1_macro')
print("Accuracy TFIDF: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))

scores = model_selection.cross_val_score(lgr,train_boolean,train_data['target'],cv = 5,scoring = 'f1_macro')
print("Accuracy Boolean: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))

#making the prediction for the test data
clf.fit(train_tfidf, train_data["target"])
target_prediction = clf.predict(test_tfidf)

#writing submission file
submission = pd.DataFrame({"id":test_data['id'], 'target':target_prediction})

#making it a csv
submission.to_csv("submission.csv",index=False)

