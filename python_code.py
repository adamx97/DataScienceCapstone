import folium
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans



print("Hello Github!")

def AddSquareToDataFrame(datadict, ctr, edges ):
    #edges = [upper_left, upper_right, lower_right, lower_left]
    NWLat = edges[0][0]
    NWLong = edges[0][1]
    NELat = edges[1][0]
    NELong = edges[1][1]
    SELat = edges[2][0]
    SELong = edges[2][1]
    SWLat =  edges[3][0]
    SWLong = edges[3][1]
    datadict[ctr] = [ctr]
    datadict[ctr].extend([NWLat, NWLong, NELat, NELong, SELat, SELong, SWLat, SWLong])
def GetSquaresToDiscard():
    squarestodiscard = set()
    squarestodiscard = squarestodiscard.union(range(21,38))
    squarestodiscard = squarestodiscard.union(range(60,76))
    squarestodiscard = squarestodiscard.union(range(98,114))
    squarestodiscard = squarestodiscard.union(range(137,152 ))
    squarestodiscard = squarestodiscard.union(range(175,190 ))
    squarestodiscard = squarestodiscard.union(range(214,228 ))
    squarestodiscard = squarestodiscard.union(range(252,266 ))
    squarestodiscard = squarestodiscard.union(range(291,304 ))
    squarestodiscard = squarestodiscard.union(range(330, 342 ))
    squarestodiscard = squarestodiscard.union(range(368,380 ))
    squarestodiscard = squarestodiscard.union(range(406,418 ))
    squarestodiscard = squarestodiscard.union(range(445,455 ))
    squarestodiscard = squarestodiscard.union(range(455,456 ))
    squarestodiscard = squarestodiscard.union(range(484,494 ))
    squarestodiscard = squarestodiscard.union(range(522,532 ))
    squarestodiscard = squarestodiscard.union(range(561, 570 ))
    squarestodiscard = squarestodiscard.union(range(600,608 ))
    squarestodiscard = squarestodiscard.union(range(638, 646 ))
    squarestodiscard = squarestodiscard.union(range(678,684 ))
    squarestodiscard = squarestodiscard.union(range(717, 722 ))
    squarestodiscard = squarestodiscard.union(range(756, 760 ))
    squarestodiscard = squarestodiscard.union(range(795, 798 ))
    squarestodiscard = squarestodiscard.union(range(834, 836 ))
    squarestodiscard = squarestodiscard.union(range(872, 874 ))
    squarestodiscard.add(911)
    squarestodiscard.add(949)
    squarestodiscard.add(1101)
    squarestodiscard = squarestodiscard.union(range(1138, 1140))    
    return squarestodiscard

baltimorecenter = (39.290389, -76.612194)

borderstr = "-76.711295892445 39.371964923459, -76.52967640072 39.371979799609, -76.52986045665 39.209630820338, -76.549727473808 39.197241314436, -76.583675299071 39.208128396535, -76.611612941603 39.234402416501, -76.711163565712 39.277846370577, -76.711295892445 39.371964923459"

tups = borderstr.split(',')
borderpoints = []
for a in tups:
    a = a.strip()
    ll = a.split(" ")
    ll
    point = (float(ll[1]), float(ll[0]))  # reverse order: long, lat  --> lat, long
    borderpoints.append(point)

xpoints = np.linspace(-76.71129589, -76.5296764, num=31 ) # 31 points east to west.
ypoints = np.linspace(39.37196492, 39.19724131, num = 39) # 39 points from north to south.and


print ("{} borderpoints: {}".format(len(borderpoints), borderpoints))
print ("{} xpoints: {}".format(len(xpoints), xpoints))
print ("{} ypoints: {}".format(len(ypoints), ypoints))


map_baltimore = folium.Map(location=baltimorecenter, zoom_start=12, control_scale=True)  # width='60%'

#bounds = [(xpoints[0], ypoints[0]), (xpoints[5], ypoints[5])];
#bounds


corner1 = (ypoints[0], xpoints[0])
corner2 = (ypoints[5], xpoints[5])

#print (bounds)
bounds = [corner1, corner2]
b = folium.FitBounds(bounds)




map_baltimore

result = pd.read_pickle('result.pkl')
print(result.head())


baltimorecenter = (39.290389, -76.612194)

borderstr = "-76.711295892445 39.371964923459, -76.52967640072 39.371979799609, -76.52986045665 39.209630820338, -76.549727473808 39.197241314436, -76.583675299071 39.208128396535, -76.611612941603 39.234402416501, -76.711163565712 39.277846370577, -76.711295892445 39.371964923459"

tups = borderstr.split(',')
borderpoints = []
for a in tups:
    a = a.strip()
    ll = a.split(" ")
    ll
    point = (float(ll[1]), float(ll[0]))  # reverse order: long, lat  --> lat, long
    borderpoints.append(point)

xlongpoints = np.linspace(-76.71129589, -76.5296764, num=31 ) # 31 points east to west.
ylatpoints = np.linspace(39.37196492, 39.19724131, num = 39) # 39 points from north to south.and


print ("{} borderpoints: {}".format(len(borderpoints), borderpoints))
print ("{} xlongpoints: {}".format(len(xlongpoints), xlongpoints))
print ("{} ylatpoints: {}".format(len(ylatpoints), ylatpoints))


yindex =3
xindex = 3

xdistance =  xlongpoints[1] - xlongpoints[0] 
xdist25 = .25 * xdistance

ydistance =  ylatpoints[1] - ylatpoints[0] 
ydist25 = .25 * ydistance
center = ((ylatpoints[yindex] + ylatpoints[yindex+1])/2.0 , (xlongpoints[xindex] + xlongpoints[xindex+1])/2.0)
location= (center[0] + ydist25, center[1] - xdist25)



#result.iloc[ctr].TrafficDriverVenueCount
ctr=3
result.CompetitorVenues
print ("drivers: {}".format(str(result.TrafficDriverVenueCount)))

print ("my venues: {} type {}".format(result.iloc[3].TrafficDriverVenueCount, type(float(result.iloc[3].TrafficDriverVenueCount * 10))))

line_color='red'
fill_color='red'
weight=1
squareDict = {}
squarestodiscard = GetSquaresToDiscard()
xdistance =  xlongpoints[1] - xlongpoints[0] 
xdist25 = .25 * xdistance

ydistance =  ylatpoints[1] - ylatpoints[0] 
ydist25 = .25 * ydistance


map_baltimore2 = folium.Map(location=baltimorecenter, zoom_start=12, control_scale=True)  # width='60%'
folium.Marker(baltimorecenter, popup='City Center').add_to(map_baltimore2)
map_baltimore2.add_child(folium.vector_layers.PolyLine(locations=borderpoints, color='blue', weight=2 ))

baltDriverLocsDf = pd.read_pickle('baltDriverLocsDf.pkl')
baltDriverLocsDf.head()

#for a in  zip (baltDriverLocsDf.Latitude.to_list(), baltDriverLocsDf.Longitude.to_list()): print (a)
#s = [[lat, lng] for lat, lng in baltDriverLocsDf.Latitude.to_list(), baltDriverLocsDf.Longitude.to_list() ]

s = list(map(list, zip (baltDriverLocsDf.Latitude.to_list(), baltDriverLocsDf.Longitude.to_list())  ))

ctr =0
while ctr <1:
    for xindex in range(len(xlongpoints)-1):
        for yindex in range(len(ylatpoints) -1):
            try:
                if ctr not in squarestodiscard:
                    upper_left=( ylatpoints[yindex], xlongpoints[xindex])
                    upper_right=( ylatpoints[yindex], xlongpoints[xindex + 1])
                    lower_right=(ylatpoints[yindex + 1], xlongpoints[xindex + 1])
                    lower_left=(ylatpoints[yindex + 1], xlongpoints[xindex])
                    edges = [upper_left, upper_right, lower_right, lower_left]
                    map_baltimore2.add_child(folium.vector_layers.Rectangle(bounds=edges, color='grey', 
                            weight=weight, html=str(ctr), popup=("square: {}".format(ctr))))
                    center = ((ylatpoints[yindex] + ylatpoints[yindex+1])/2.0 , (xlongpoints[xindex] + xlongpoints[xindex+1])/2.0)
                    if result.loc[ctr].TrafficDriverVenueCount > 0:
                        myrad = result.loc[ctr].TrafficDriverVenueCount * 5.0  # result.loc[ctr].TrafficDriverVenueCount *5
                        line_color = 'blue'
                        map_baltimore2.add_child(folium.vector_layers.Circle(
                                                    location= (center[0] + ydist25, center[1] - xdist25), color=line_color, fill_color=line_color,
                                                    radius = result.loc[ctr].TrafficDriverVenueCount * 5,
                                                    #radius = myrad,
                                                    weight=weight, html=str(ctr), 
                                                    popup=("drivers: {}".format(str(result.loc[ctr].TrafficDriverVenueCount)))
                                                ))
                    if result.loc[ctr].CompetitorVenueCount > 0: 
                        line_color = 'red'
                        map_baltimore2.add_child(folium.vector_layers.Circle(location= (center[0] - ydist25, center[1] + xdist25), color=line_color, fill_color=line_color,
                    #                                 radius = result.loc[ctr].CompetitorVenueCount * 5,
                                                    radius = 50,
                                                    weight=weight, html=str(ctr), popup=("compet: {}".format(str(result.loc[ctr].CompetitorVenueCount)))))
            except Exception as ex:
                print ("Except: ctr= {} {}".format(ctr, ex))
            finally:
                ctr += 1

#map_baltimore2             

def dftester(dataframe):
    dataframe.head()


def kmeanstests():
    number_of_clusters = 5
    df_viable_squares = pd.read_pickle('df_viable_squares.pkl')
    viable_latlng = df_viable_squares[['NWLat', 'NWLong', 'NELat', 'NELong', 'SELat', 'SELong', 'SWLat', 'SWLong']].values
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(viable_latlng)
    kmeans.cluster_centers_
    ctr = 0
    for index, row in df_viable_squares.iterrows():
        print ("Index: {} df_viable_squares squareid [index]: {} kmeans.labels_[{}] = {}".format(ctr, df_viable_squares.SquareId[index], index, kmeans.labels_[ctr]) )
        ctr +=1

def hdbtest():
    import hdbscan
    driverlatlng = baltDriverLocsDf[['Latitude', 'Longitude']].values
    driverlatlngRad = np.radians(driverlatlng) #convert the list of lat/lon coordinates to radians
    earth_radius_km = 6371

    epsilon = 1/ earth_radius_km #calculate 250 meter epsilon threshold

    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='haversine',
    cluster_selection_epsilon=epsilon, cluster_selection_method = 'eom')
    hdb_td_clusters = clusterer.fit(driverlatlngRad)
    print (clusterer.labels_)

# Great circle method for calculating distance based on Latitude and Longitude.  It can be found many places on the internet.
from math import radians, cos, sin, asin, sqrt 
def distance(lat1, lat2, lon1, lon2): 
      
    # The math module contains a function named 
    # radians which converts from degrees to radians. 
    lon1 = radians(lon1) 
    lon2 = radians(lon2) 
    lat1 = radians(lat1) 
    lat2 = radians(lat2) 
       
    # Haversine formula  
    dlon = lon2 - lon1  
    dlat = lat2 - lat1 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  
    c = 2 * asin(sqrt(a))  
     
    # Radius of earth in kilometers. Use 3956 for miles 
    r = 6371
       
    # calculate the result 
    return(c * r) 

if __name__ == '__main__':
    #map_baltimore2
    # driver code  
    lat1 = 53.32055555555556
    lat2 = 53.31861111111111
    lon1 = -1.7297222222222221
    lon2 =  -1.6997222222222223
    print(distance(lat1, lat2, lon1, lon2), " km") 
    data = pd.read_pickle('data.pkl')
    dftester(data)
    kmeanstests()
    hdbtest()

    print ("counting ", end ='')
    for a in range (1,100):
        print (". ", end="")
    print ("done")
    print ("new line")