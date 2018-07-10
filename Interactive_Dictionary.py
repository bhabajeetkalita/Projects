'''
Author: Bhabajeet Kalita
Description: A program to return the definition of a word in english by comparing the word we entered with a bag of words in a json file
'''
import json
from difflib import get_close_matches

data = json.load(open("/Users/Gourhari/Documents/Job/Git Hub/Python/Interactive Dictionary/Interactive_Dictionary_Bag_of_words.json"))

def translate(w):
    #Change all to lowercase
    w = w.lower()
    if w in data:
        return data[w]
    elif len(get_close_matches(w, data.keys())) > 0:
        yn = input("\n Hey, did you mean %s instead? Enter Y if yes, or N if no: " % get_close_matches(w, data.keys())[0])
        if yn == "Y":
            return data[get_close_matches(w, data.keys())[0]]
        elif yn == "N":
            return "\n The word doesn't exist. Please double check it once again. Thank you!"
        else:
            return "\n Oops!!!!! We didn't understand your entry. Sorry!"
    else:
        return "\n The word doesn't exist. Please double check it once again. Thank you!"

word = input("Please enter the word that you want to search: ")
output = translate(word)
print(word+ ": means: ")
if type(output) == list:
    for item in output:
        print("\n"+item)
else:
    print(output)
