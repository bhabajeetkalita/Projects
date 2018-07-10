'''
Author: Bhabajeet Kalita
Date: 30 - 11 - 2017
Description: Demonstration of Multi Index using Pandas
'''
import pandas as pd

bigmac = pd.read_csv("bigmac.csv",parse_dates=["Date"]) #Date is converted into date-type
bigmac.dtypes

#Create a multi index using .set_index() method

bigmac.set_index(keys=["Date","Country"],inplace=True)
#OR Reverse it
#bigmac.set_index(keys=["Country","Date"])
bigmac.sort_index() #Sort both C & D in ascending order
bigmac.index.names
type(bigmac.index)
bigmac.index[0]
bigmac.index[1]

#.get_level_values() method

bigmac = pd.read_csv("bigmac.csv",parse_dates=["Date"],index_col=["Date","Country"])
bigmac.sort_index(inplace=True)
bigmac.index #Multi-index object
bigmac.index.get_level_values(0) #Only 1st level - dates
bigmac.index.get_level_values(1) #Only countries
bigmac.index.get_level_values("Date")

#.set_names() method - replace column names
bigmac = pd.read_csv("bigmac.csv",parse_dates=["Date"],index_col=["Date","Country"])
bigmac.sort_index(inplace=True)
bigmac.index.set_names(["Day","Location"],inplace=True) #Change dates to day and country to location

#.sort_index() method - sort both columns of title into asc/des order
bigmac = pd.read_csv("bigmac.csv",parse_dates=["Date"],index_col=["Date","Country"])
bigmac.sort_index(ascending=[True,False],inplace=True)
bigmac.head()

#Extract rows from a multi index DF
'''
bigmac = pd.read_csv("bigmac.csv",parse_dates=["Date"],index_col=["Date","Country"])
bigmac.sort_index(inplace=True)
bigmac.loc[("2010-01-01","Brazil"),"Price in US Dollars"] #Need values from both the indexes - list doesn't work - pass tuple (immutable - can't be modified after creation)
bigmac.loc[("2015-07-01","Chile"),"Price in US Dollars"]
bigmac.ix[("2016-01-01","China"),0]
'''

#.transpose method
'''
bigmac = pd.read_csv("bigmac.csv",parse_dates=["Date"],index_col=["Date","Country"])
bigmac.sort_index(inplace=True)
bigmac = bigmac.transpose()
bigmac.ix["Price in US Dollars",("2016-01-01","Denmark")]
'''
#swaplevel() method - swaps the levels within a multi-level index
bigmac = pd.read_csv("bigmac.csv",parse_dates=["Date"],index_col=["Date","Country"])
bigmac.sort_index(inplace=True)
bigmac = bigmac.swaplevel()

#.stack() method - takes columns and moves it together
world = pd.read_csv("worldstats.csv",index_col=["country","year"])
s = world.stack()
#s.unstack().unstack().unstack()

#.unstack() method
world = pd.read_csv("worldstats.csv",index_col=["country","year"])
#s.unstack()
s=world.stack()
#s.unstack(2) #Move both values into columns again
#s.unstack("country")
s.unstack(level=[1,0])
s.unstack(level=["year","country"])
s = s.unstack("year",fill_value = 0) #fill all Na values


#.pivot() method - make values into columns - only for small unique values
sales = pd.read_csv("salesmen.csv",parse_dates=["Date"])
sales["Salesman"] = sales["Salesman"].astype("category")
sales.pivot(index="Date",columns="Salesman",values="Revenue")

#pivot_table() - aggregate values as a whole based on the unique values of a column
foods = pd.read_csv("foods.csv")
foods.pivot_table(values="Spend",index="Item",aggfunc="sum")
foods.pivot_table(values="Spend",index=["Gender","Item"],aggfunc="sum")
foods.pivot_table(values="Spend",index=["Gender","Item"],columns=["Frequency","City"],aggfunc="sum")
#pd.pivot_table(data=foods,values="Spend",index=["Gender","Item"],columns=["Frequency","City"],aggfunc="sum")


#pd.melt() method - reverse of .pivot_table
sales = pd.read_csv("quarters.csv")
sales
pd.melt(sales,id_vars="Salesman")
pd.melt(sales,id_vars="Salesman",var_name="Quarter",value_name="Revenue")


#Section:8 GroupBy
#110. Intro to the Grouby Module
#Ideally group by values which have duplicate values - Sector example here
import pandas as pd
fortune = pd.read_csv("fortune1000.csv",index_col="Rank")
sectors = fortune.groupby("Sector")
type(sectors)
fortune.head(3)

#111. Operations of groupby

len(sectors) #No. of unique groupings
fortune["Sector"].nunique() == len(sectors)
sectors.size() #How many rows fall into each groups
fortune["Sector"].value_counts()

sectors.first()#First row of every group

sectors.last()

sectors.groups #Dictionary { unique groups : list of all rows that fall into that group}
fortune.loc[24]

#112. Retrieve a group using .get_group() method
fortune = pd.read_csv("fortune1000.csv",index_col="Rank")
sectors = fortune.groupby("Sector")
sectors.get_group("Energy")

#113. Methods on Groupby object and dataframe columns
fortune = pd.read_csv("fortune1000.csv",index_col="Rank")
sectors = fortune.groupby("Sector")
sectors.max() #look at the left most column and arrange accordingly max or min
sectors.min()
sectors.sum()
sectors.mean()

sectors["Revenue"].sum()
sectors["Profits"].min() #Worst performing company

sectors[["Revenue","Profits"]].sum()

#114. Grouping by multiple columns
fortune = pd.read_csv("fortune1000.csv",index_col="Rank")
sectors = fortune.groupby(["Sector","Industry"])
sectors.size()
sectors.sum()
sectors["Revenue"].sum()
sectors["Employees"].mean()

#115. The .agg() Method - allows to call different aggregated values
fortune = pd.read_csv("fortune1000.csv",index_col="Rank")
sectors = fortune.groupby("Sector")
sectors.agg({"Revenue":"sum",
             "Profits":"sum",
             "Employees":"mean"})

sectors.agg(["size","sum","mean"])

sectors.agg({"Revenue":["sum","mean"],
             "Profits":"sum",
             "Employees":"mean"})

#116. Iterating through the groups
fortune = pd.read_csv("fortune1000.csv",index_col="Rank")
sectors = fortune.groupby("Sector")

df=pd.DataFrame(columns=fortune.columns)
df
for sector,data in sectors:
    highest_revenue_company_in_group = data.nlargest(1,"Revenue")
    df = df.append(highest_revenue_company_in_group)

cities = fortune.groupby("Location")
df=pd.DataFrame(columns=fortune.columns)
df

for city,data in cities:
    highest_revenue_in_city = data.nlargest(1,"Revenue")
    df = df.append(highest_revenue_in_city)

df
