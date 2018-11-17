import os
import numpy as np
import pandas as pd
import shapefile
from matplotlib import pyplot as plt
baseDir = 'data/california/california/train'

# meta_data = pd.read_csv(os.path.join(baseDir,'BG_METADATA_2016.csv'))
# race_data = pd.read_csv(os.path.join(baseDir,'X02_RACE.csv'))

ca_sf = shapefile.Reader('data/cb_2017_06_tract_500k/cb_2017_06_tract_500k')
print(len(ca_sf.shapes()))
for i in range(len(ca_sf.shapes())):
    ca_shape = ca_sf.shape(i)
    x_data = np.zeros((len(ca_shape.points),1))
    y_data = np.zeros((len(ca_shape.points),1))
    for point in range(len(ca_shape.points)):
        x_data[point] = ca_shape.points[point][0]
        y_data[point] = ca_shape.points[point][1]

    plt.plot(x_data,y_data,'k')

plt.show()
# print(race_data['GEOID'])