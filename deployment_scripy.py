import streamlit as st 
import warnings 
import pandas as pd 
import numpy as np  
from tweepy import OAuthHandler
import re 
from wordcloud import WordCloud, ImageColorGenerator
import openpyxl
import time
import tqdm
import json 
import matplotlib.pyplot as plt
import joblib
import tweepy
import twint


st.set_option("deprecation.showfileUploaderEncoding", False)
st.set_option("deprecation.showPyplotGlobalUse", False)

STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """

def main():
    html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:40px;padding:9px">Live twitter Sentiment analysis</p></div>
	"""
    st.title("Brand Preference Indicator")
    st.markdown(html_temp, unsafe_allow_html=True)
    
    #API KEYS
    cons_key = "z1pFm2CMlualoHl6e4Runo9mp"
    cons_secret = "x5X6hAbmEuUov6f7O8kiltwECZ2oFXlnsUlUBzEHXSAE8tVPVw"
    access_tok = "1336751394070654982-69h5b6wJ14IqvLTprnTqMjeZzIS3Dq"
    access_tok_sec = "IFCWcjkNpdF1cfoyiIkKAjzGa7Kf2QJBXJzZ4Q3jNN3sE"

    ##Authentication
    auth = tweepy.OAuthHandler(cons_key, cons_secret)
    auth.set_access_token(access_tok, access_tok_sec)

    api = tweepy.API(auth)
    #creating a dataframe to store our data
    df = pd.DataFrame(columns=["Date","User", "Tweet", "LikeCount", "RetweetCount"])
    #Quick function that returns tweets for the brand name searched:
    #For now, accepted brand names are airtel and/or safaricom
    def get_brand_tweets(Brand, Count):
        i = 0
        for tweet in tweepy.Cursor(api.search_tweets, q=Brand, count=100, lang="en", exclude="retweets").items():
            df.loc[i, "Date"] = tweet.created_at
            df.loc[i, "User"] = tweet.user.name
            df.loc[i, "Tweet"] = tweet.text
            df.loc[i, "LikeCount"] = tweet.favorite_count
            df.loc[i, "RetweetCount"] = tweet.retweet_count
            i = i + 1
            if i >= Count:
                break
            else:
                pass
    def clean_tweets(tweets):
        '''
        A function that cleans tweets from their messy raw form to cleaner texts that can be encoded for subsequent preprocessing
        '''
        #stopwords = nltk.corpus.stopwords.words('english')
        newStopWords = ("kwanza", "hello", "hi", "kwa", "hey", 
               "ni", "si","na","tu", "za", "yake")
        #stopwords.extend(newStopWords)
        #removing extra spaces
        regex_pat = re.compile(r'\s+')
        tweets = tweets.str.replace(regex_pat, ' ')
        # removal of @name[mention]
        regex_pat = re.compile(r'@[\w\-]+')
        tweets = tweets.str.replace(regex_pat, '')
        # removal of links[https://abc.com]
        giant_url_regex =  re.compile(r"http\S+")
        tweets = tweets.str.replace(giant_url_regex, '')
        #removing stopwords
        tweets = tweets.apply(lambda x: ' '.join([word for word in x.split() if word not in (newStopWords)]))
        #removal of punctuations and numbers
        #tweets = tweets.str.replace("[^a-zA-Z]", " ")
        # removal of capitalization
        #tweets = tweets.str.lower()
        return tweets
    def load_model(model_path):
        clf = joblib.load(model_path)
        return clf
    
    Brand = str()
    Brand = str(st.text_input("Enter brand to be analyzed: "))

    if len(Brand) > 0:
        with st.spinner("Extracting tweets, please wait..."):
            get_brand_tweets(Brand, Count=200)
        
        st.success("Successful tweet extraction!")

    df['clean_tweets'] = clean_tweets(df['Tweet'])
    df['Predicted_sentiments'] = load_model("saf_retrained_model.sav").predict(df['clean_tweets'])
    def get_num_sentiments(data):
        pos_tweets = len(data[data['Predicted_sentiments'] == "positive"])
        neg_tweets = len(data[data['Predicted_sentiments'] == "negative"])
        neut_tweets = len(data[data['Predicted_sentiments'] == "neutral"])
        return pos_tweets, neg_tweets, neut_tweets
    pos, neg, neut = get_num_sentiments(df)

    #plotting value count per sentiment
    st.subheader("Distribution of sentiments in today's data")
    st.write(df['Predicted_sentiments'].value_counts().plot(kind='barh'))
    st.pyplot()
    #Printing summary
    st.write("Number of positive tweets: " +str(pos))
    st.write("Number of negative tweets: " +str(neg))
    st.write("Number of neutral tweets: " +str(neut))

    if st.button("Exit"):
        st.balloons()

if __name__ == "__main__":
    main() 