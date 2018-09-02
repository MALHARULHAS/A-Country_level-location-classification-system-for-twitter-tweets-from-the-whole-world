# Country_level-location-classification-system-for-Twitter-tweets-of-the-whole-world

Description:
------------
Our main motive was to develop a Countrylevel location classification system for worldwide tweets. Our work is inspired from the existing work done by (see links: https://arxiv.org/pdf/1604.07236.pdf, https://github.com/azubiaga/tweet-country-classification). Our small contribution involves the use of user friends network feature along with all other features used in the existing work.

There are two sample files of datasets one is for training the model & other for testing. The datasets are unbalanced since only a sample of them are provided. The python code file contains the code for the 9 classifiers which we used. There is a ppt presentation which might give you a good outlook to this work. 

The countries represented are given in form of indexes 0 to 248. 300 is no country identified. 
see file "modified_country_list.txt".
The country index starts from 0 assigned to the first country in the above file. Each country is spelt deifferently hence somany different names for the same country.

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
