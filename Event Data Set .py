#!/usr/bin/env python
# coding: utf-8

# # Event Prediction - Assignment  
# 
# ## In that notebook 
# 
# ### - I run all the task which are given 
# #### Event Types Distribution: Analyze the distribution of event types to understand the frequency of different meteorological events. You can use visualizations like bar plots or pie charts to display this information.
# 
# Temporal Patterns: Explore temporal patterns by analyzing the number of events over different time periods (e.g., by year, month, or day). This can help identify any trends or seasonal variations in meteorological events.
# 
# Geographical Distribution: Visualize the geographical distribution of events on a map to identify regions that are more prone to certain types of meteorological events. This can be done using tools like GeoPandas or Folium.
# 
# Spatial Clustering Analysis: Conduct spatial clustering analysis to identify clusters of events and understand the spatial distribution of high-risk zones. This can help in identifying areas that are more susceptible to multiple meteorological events.
# 
# Event Severity Analysis: Analyze the severity of different event types by considering factors such as the number of occurrences, duration, and impact on the region. This can provide insights into the most severe meteorological events.
# 
# Yearly Analysis: Compare the number of events across different years to identify any significant variations or trends over time. This can help in understanding the long-term patterns of meteorological events.
# 
# Correlation Analysis: Explore correlations between different variables in the dataset (e.g., event type, location, time) to identify any relationships or dependencies. This can provide insights into the factors that influence the occurrence of meteorological events.
# 
# Events Within Radius: Calculate the count of total events occurring within a specified radius of a point of interest. This can help in understanding the local impact of meteorological events on specific regions.
# 
# Summary of Findings: Summarize the key insights and findings from your analysis, highlighting any significant patterns, trends, or correlations observed in the dataset.

# In[1]:


# importing 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[2]:


# warning 
import warnings 
warnings.filterwarnings('ignore')


# In[3]:


# data read 
event = pd.read_csv('D:\chrome\events.csv')


# In[4]:


# starting 5 rows and columns // " Exploratory data analysis "
event.head()


# In[5]:


print(event.info())


# In[6]:


event.shape


# In[7]:


event.index


# In[8]:


event.columns


# In[9]:


event.dtypes


# In[10]:


event['event-type'].unique()


# In[11]:


event.count()


# In[12]:


event['event-type'].value_counts()


# In[13]:


event.info


# In[ ]:


#import 
from shapely.geometry import box 
from scipy.spatial import cKDTree as KDTree #for inverse Distance Weight calculation


plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


event.describe()


# In[ ]:


event.isnull().sum()


# In[ ]:


event['city'].fillna('Unknown City', inplace=True)
event['county'].fillna('Unknown county', inplace=True)


# In[ ]:


print(event.isnull().sum())


# In[ ]:


# Explore distribution of event types
plt.figure(figsize=(10, 6))
event['event-type'].value_counts().plot(kind='bar')
plt.title('Distribution of Event Types')
plt.xlabel('Event Type')
plt.ylabel('Frequency')
plt.show()


# ### In distribution of event Heavy Rain event is showing 200000 Frequency and Hurricane having aprox 0 to 100 and storme surge is not showing nothing¶

# In[ ]:


# Explore temporal trends
event['date'] = pd.to_datetime(event['date'])
event['year'] = event['date'].dt.year
plt.figure(figsize=(10, 6))
event['year'].value_counts().sort_index().plot(kind='line')
plt.title('Number of Events by Year')
plt.xlabel('Year')
plt.ylabel('Number of Events')
plt.show()


# ### Acording to data temporal trends showing heighest no of event 2017 - 2020 = 35000

# In[ ]:


import geopandas as gpd
from shapely.geometry import Point
import os
os.environ['GDAL_DATA'] = '/path/to/gdal_data_directory'


# Create GeoDataFrame from latitude and longitude
gdf = gpd.GeoDataFrame(event, geometry=gpd.points_from_xy(event.longitude, event.latitude))

# Plot distribution of events on a map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(figsize=(10, 6))
gdf.plot(ax=ax, color='red', markersize=5)
plt.title('Distribution of Weather Events')
plt.show()

# Calculate number of events per city
events_per_city = event['city'].value_counts()

# Calculate number of events per county
events_per_county = event['county'].value_counts()

# Calculate number of events per state
events_per_state = event['state'].value_counts()


# ### Acording to data Distribution of weather Events is spotted Red color area

# In[ ]:


# Classify the data based on the Year and conclude which year is most affected
event['year'] = event['date'].dt.year
yearly_counts = event['year'].value_counts()
most_affected_year = yearly_counts.idxmax()
print(f"The most affected year is: {most_affected_year}")


# In[ ]:


# Create a heatmap using seaborn
plt.figure(figsize=(6, 4))
sns.kdeplot(data=event, x='longitude', y='latitude', cmap='viridis', fill=True, thresh=0, levels=100)

# Add a title and labels
plt.title('Heatmap of Weather Events')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Show the heatmap
plt.show()


# In[ ]:


# Define severity levels for event types
severity_levels = {
    'Heavy Rain': 'Moderate',
    'Flood': 'Severe',
    'Flash Flood': 'Severe',
    'Hurricane': 'Severe',
    'Storm Surge': 'Severe',
    'Coastal Flood': 'Moderate'
}

# Map severity levels to event types
event['severity'] = event['event-type'].map(severity_levels)

# Plot distribution of severity levels
plt.figure(figsize=(8, 5))
event['severity'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Severity Levels')
plt.xlabel('Severity Level')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


# Check DataFrame 'event'

import matplotlib.pyplot as plt
    # Plot distribution of severity levels
plt.figure(figsize=(8, 5))
event['event-type'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of event-type Levels')
plt.xlabel('event-type')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


import folium

# Create a map centered on a specific location (e.g., USA)
m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)

# Add markers for each weather event
for idx, row in event.iterrows():
    folium.Marker([row['latitude'], row['longitude']], popup=row['event-type']).add_to(m)

# Display the map
m


# In[ ]:


from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius = 6371  # Radius of the Earth in kilometers
    distance = radius * c
    return distance

def count_events_within_radius(latitude, longitude, radius_km, event):
    """
    Calculate the count of total events occurring within a specified radius (in kilometers) of a point.

    Parameters:
    - latitude (float): Latitude coordinate of the point.
    - longitude (float): Longitude coordinate of the point.
    - radius_km (float): Radius in kilometers within which to count events.
    - event (DataFrame): DataFrame containing the weather events data.

    Returns:
    - event_count (int): Count of events occurring within the specified radius of the point.
    """
    event_count = 0
    for idx, row in event.iterrows():
        event_lat = row['latitude']
        event_lon = row['longitude']
        distance = haversine(latitude, longitude, event_lat, event_lon)
        if distance <= radius_km:
            event_count += 1
    return event_count

# Example usage:
latitude = 40.7128  # Latitude of the point
longitude = -74.0060  # Longitude of the point
radius_km = 50  # Radius in kilometers

# Call the function to count events within the specified radius of the point
total_events_count = count_events_within_radius(latitude, longitude, radius_km, event)
print(f'Total events within {radius_km} KM of the point: {tot


#                                  ### Asignment from Yash Pratap Singh                                                                                                                  
#        
