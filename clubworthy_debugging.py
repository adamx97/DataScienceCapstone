# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Club Worthy: Finding a neighborhood suitable for a live music venue
# %% [markdown]
# ## Introduction/Business Problem
# Many cities seem to have too few live music venues.  This project will attempt to find a good neighborhood for a music venue.  To be successful, the location should not already have too many music venues, but have a sufficiently robust nightlife to support another night time business.  We will look for neighborhoods with sufficient restaurants, bars, cafes, and dance or night clubs (which tend to have recorded music) but with fewer live music venues.  If a neighbordhood has no restarants or bars, it probably isn't a place that people frequent at night for entertainments, and it seems less likely that people out for the evening would want to make a special trip just for one venue.  On the other hand, with sufficient bars and restaurants, the neighborhood would be enhanced by an additional nighttime venue and help add to the overall traffic.  If an area can be found within a reasonable distance of two nightlife centers, that might be optimal, since it could help bridge the two and benefit from the traffic at both.  This data would be valuable to a business owner who wanted to start a new music venue.
# %% [markdown]
# ## Data 
# We will use Foursquare venue location data to find clusters of bars, restaurants, clubs and other music venues within the city and determine where they are and where they are missing.  We will try to find ideal locations amongst other existing nightlife destinations that aren't already served by music venues.  A future revision to this  criteria for this exercise would be to include the demographics of the city, to attempt to determine if the residents were in likely age groups, and had sufficient disposable income.

# %%
CLIENT_ID = 'C3NLZY0MJGCJENLKP4I5R0NONFIS3A4DPRNLGV33MWAU5Y3O' # your Foursquare ID
CLIENT_SECRET = '2OOM4WNGBAPZ5M0NC0E4JF31EZXS2KD2XQETJVHAIGTOWMZM' # your Foursquare Secret
VERSION = '20180604'
LIMIT = 30
print('Your credentails are hidden as CLIENT_ID, CLIENT_SECRET.')
#print('CLIENT_ID: ' + CLIENT_ID)
#print('CLIENT_SECRET:' + CLIENT_SECRET)


# %%
# imports
# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering librar
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
import numpy as np # library to handle data in a vectorized manner
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
from IPython.display import HTML, display
import geocoder # import geocoder
import pandas as pd

# %% [markdown]
# ## Methodology
# 
# ### Super simple zip code data wrangling
# Let's start by collecting venues in Baltimore, our target city.  A quick internet search finds: http://www.ciclt.net/sn/clt/capitolimpact/gw_ziplist.aspx?ClientCode=capitolimpact&State=md&StName=Maryland&StFIPS=&FIPS=24510 
# 
# Two minutes with Excel and notepad produces the Baltimore city zip code list:
# 21201, 21202, 21203, 21205, 21206, 21209, 21210, 21211, 21212, 21213, 21214, 21215, 21216, 21217, 21218, 21223, 21224, 21225, 21229, 21230, 21231, 21233, 21239, 21270, 21279, 21281, 21297
# 
# Some further investigation shows some of these zip codes extend from the city into the surrounding counties. https://www.zipdatamaps.com/zipcodes-baltimore-md The zip codes that are largely outside the city as shown in the zip code map are not found in the list of zip codes we show above.
# 
# 
# <img src="BaltimoreCity.png" height='400' alt="Baltimore City map">
# <img src="BaltimoreZipCodes.png" height='400' alt="Baltimore City map">
# 
# Foursquare provides the ability to provide a rectangular bounding box, we could use that for the larger, Northern, very rectangular part of the city, and then attempt to add on the irregularly Sourthern  For the Southern, irregularly part shaped of the county, we'll use the sourthern zip codes, and accept that some of the venues may be outside the city.  We expect fewer venues in the Sourther parts, so we can evaluate that impact as we go.  Foursquare also allows us to examine a circular area of a given radius. 
# 
# 
# For now, we'll stick with the zip codes.   We may find interesting candidate sites just outside the city, which would be considered a bonus.
#     
# 
# 
# %% [markdown]
# ## Use Foursquare to retrieve venues from the zip codes
# 
# 
# 
# 
# Find the latitude and longitude centers of each zip code using geocoder.  

# %%
#From https://data.baltimorecity.gov/Geographic/Baltimore-City-Line/rz8b-wbi9 we get the administrative border of Baltimore city that can be exported in the form of longitude, #latitude tuples.

borderstr = "-76.711295892445 39.371964923459, -76.52967640072 39.371979799609, -76.52986045665 39.209630820338, -76.549727473808 39.197241314436, -76.583675299071 39.208128396535, -76.611612941603 39.234402416501, -76.711163565712 39.277846370577, -76.711295892445 39.371964923459"

tups = borderstr.split(',')
borderpoints = []
for a in tups:
    ll = a.split(" ")
    ll
    point = (float(ll[1]), float(ll[0]))  # reverse order: long, lat  --> lat, long
    borderpoints.append(point)

borderpoints





# %%
BaltimoreZips = [21201, 21202, 21203, 21205, 21206, 21209, 21210, 21211, 21212, 21213, 21214, 21215, 21216, 21217, 21218, 21223, 21224, 21225, 21229, 21230, 21231, 21233, 21239, 21270, 21279, 21281, 21297]
def getZipLatitudeLongitude(city, ziplist):
    cleaned = {}
    for postal_code in ziplist:
        # initialize your variable to None
        lat_lng_coords = None
        ctr = 0
        # loop until you get the coordinates
        while(lat_lng_coords is None):
            ctr +=1
            g = geocoder.google('{}, {}'.format(postal_code, city))
            lat_lng_coords = g.latlng
        #print ("Got {} after {} tries".format(postal_code, ctr))

        latitude = lat_lng_coords[0]
        longitude = lat_lng_coords[1]
        cleaned[postal_code]= [postal_code]
        cleaned[postal_code].append(latitude)
        cleaned[postal_code].append(longitude)
    return cleaned

cleaned = getZipLatitudeLongitude("Baltimore, Maryland", BaltimoreZips)
t_headers = ['Zipcode', 'Latitude', 'Longitude']
data = pd.DataFrame.from_dict(cleaned, orient='index', columns=t_headers)
data.head()


# %%
#baltimore lat/long : 39.2904° N, 76.6122° W aka 39.2904,-76.6122

FoursquareCategories = { '4d4b7104d754a06370d81259':'Arts & Entertainment', '4d4b7105d754a06374d81259': 'Food', '4d4b7105d754a06376d81259': 'Nightlife Spot' }
catTrafficDrivers ={'4d4b7105d754a06374d81259': 'Food'}
catCompetitors =  { '4d4b7104d754a06370d81259':'Arts & Entertainment', '4d4b7105d754a06376d81259': 'Nightlife Spot' }

def getNearbyVenues(city, zipcodes, latitudes, longitudes, categorylist, radius=500):
    venues_list=[]
    categorystring = ','.join(categorylist)
    for zipcode, lat, lng in zip(zipcodes, latitudes, longitudes):
        print(zipcode)
            
        # create the API request URL
        #url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&categoryId={}&ll={},{}&zip={}&limit={}'.format(            
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            categorystring,
            zipcode, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)

baltimore_trafficdrivers = getNearbyVenues(catTrafficDrivers, 
                                   BaltimoreZips,
                                   longitudes=baltimore_subset['Longitude'], radius=600
                                  )

