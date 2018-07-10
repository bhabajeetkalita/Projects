'''
Author: Bhabajeet Kalita
Date: 28 - 11 - 2017
Description: Parse cities, countries and dates from a given json file and rank the rows on the basis of the quality of output and transfer the rows back to a json file
'''

#Importing the relevant modules needed for this exercise - Pandas and geotext
import pandas as pd

from geotext import GeoText

#Reading the json file into a pandas dataframe
df = pd.read_json('/Users/Gourhari/Desktop/parsing_challenge.json')

#Naming the imported column as Data
df.columns = ['Data']

#Observing the first 5 rows of the dataframe
df.head()

#Creating 2 lists to store the cities and countries

cities = []

countries = []

#Iterating through the values of the 'Data' column
for row in df['Data'].values.T.tolist():

    #Append the city name by parsing city names using GeoText from the row to the cities list
    cities.append(''.join(GeoText(row).cities))

    #Append the country name by parsing country names using GeoText from the row to the countries list
    countries.append(''.join(GeoText(row).countries))

#Assigning the lists to two new columns City and Country
df['City'] = cities

df['Country'] = countries

#Creating a new column Parsing the dates from the 'Data' column using standard pandas to_datetime function

#Displaying NaT for errors using coerce

df['Date'] = pd.to_datetime(df['Data'],exact=True,errors='coerce')

#Creating a new rank column to rank the rows based on the quality of their output

#Rank = length of the City name of the row + length of the Country name of the row + 10 (if there exists a date) or 0 otherwise

df['Rank'] = df['City'].str.len() + df['Country'].str.len() + df['Date'].apply(lambda row : 0 if str(row) == "NaT" else 10)

#Filling the NaT values of the 'Date' Column with "-" for a cleaner output

df['Date'] = df['Date'].fillna("-")

#Transfering the data frame by rows to a json file in the desktop
df.to_json('/Users/Gourhari/Desktop/Bhabajeet_solutions.json',orient = "records")

'''
Explanation :
1. We import the necessary modules needed for performing the data analysis.

2. We perform some exploratory data analysis using the head() function after importing the json file to a pandas dataframe.

3. We use Pandas Library to perform the operations because is very easy to use and understand and work on tabular data.

4. Using the geotext module, we parse the city and country names from the rows of the dataframe because it is quite fast and does a reasonably good job.

5. Various other approaches could be used to parse city and country names -
    a. Named Entity Recognition using the Stanford Named Entity Recogniser in NLTK module of python (Natural Language Toolkit)
    I tried writing the following code, but it was taking too long to process the "L" - Location tags and so, I used geotext instead

import nltk
from nltk.corpus import stopwords
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import pandas as pd
from itertools import groupby

st = StanfordNERTagger('/Users/Gourhari/Documents/Studies/Master Thesis/Web-front/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz','/Users/Gourhari/Documents/Studies/Master Thesis/Web-front/stanford-ner-2014-06-16/stanford-ner.jar',encoding='utf-8')

df = pd.read_json('/Users/Gourhari/Desktop/parsing_challenge.json')
df.columns = ['Data']
locations = []
for data in df['Data']:
    classified_text = st.tag(data)
    for tag, chunk in groupby(classified_text, lambda x:x[1]):
        if tag == "L":
            locations.append(str(chunk))
        else:
            locations.append("")

    b. geograpy python package - had import issues after several attempts of installation of the other underlying packages
    c. Using self made regular expressions

6. We parse the dates from the dataset using the standard Pandas to_datetime method and store it in another column Dates

7. Various other approaches can also be used -
    a. datetime.datetime.strptime("01-Jan-1995", "%d-%b-%Y")
    b. dateutil.parser.parse('01-Jan-1995').date()

8. After trying for all possible options to parse dates, I felt the default to_datetime method yielded good results.

9. For the next task of ranking the rows based on the quality of their output, different formulas can be used. I used the following -
    a. Give 10 points to a row if there is a date present. If there is no date, then give 0 points.
    b. Calculate the length of the city name and the country name for every row and add both the values.
    c. Add all the 3 values for city, country and date and set the ranking.

10. Output the dataframe to a json file based on records (rows).

Tradeoffs -
1. Some country names like United States of America could not be parsed, since in the geotext database, only United States is mapped to US.
2. Names like Saarcbrucken & Dezember could not be parsed since it is in Germany/French.

Future Advancements -
1. Some very strong self made regular expresssions for parsing such data.
2. Multi lingual parsing.


Thank you very much.

Regards,
Bhabajeet Kalita.
'''
