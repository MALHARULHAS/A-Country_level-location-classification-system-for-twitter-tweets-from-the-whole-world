# Country_level-location-classification-system-for-Twitter-tweets-of-the-whole-world

Description

This contains some datasets &amp; classification code.

The countries represented are given in form of indexes 0 to 248. 300 is no country defiined. see file modified_country_list.txt.The country index starts from 0 assigned to the first country in the above file. Each country is spelt deifferently hence somany different names for the same country.

Each coloumn in the datasets refer place, coordinates, Timezone, user location field location indicative words, user location field organisation indicative words, USERNAME, user Text content, user Description content, user interface Language,user Tweet Language, maximum user friends location. The last coloum is the class which is the country index it predicted.

Note:-
The downloading of tweets from the Twitter API, preprocessing & conditioning of the tweets are two prelimnary steps which are done to generate the above datasets. I intend to make those codes open in future. Still downloading of tweets from Twitter streaming API is demonstrated best by see link: https://www.youtube.com/watch?v=pUUxmvvl2FE. 

# How To Run The Code

I have use Python 3.6 Spyder IDE in Anaconda Navigator.

To run this code you just have to bring this code in to the SPYDER IDE.

Now there are a total of nine classifiers given in the code, to run each one just remove the (''') apostrophes (the starting & the ending ones for each classifier code). 

Once you are done training & testing using one classifier and then you want to use another put back the apostrophes to the former one and remove the apostrophes of the next classifier code which you might want to run.

For every code you run it generates the confusion matrix and entire summary see it in the “confusion_matrix_imgs” folder in the “MALHAR_IMPLEMENTATION_CODE” directory.

You can find the documentation page attached.
