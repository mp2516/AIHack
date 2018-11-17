import os, ast
import numpy as np
import pandas as pd
from tqdm import trange
import math
import shapefile
from matplotlib import pyplot as plt
from sklearn.cluster.k_means_ import KMeans
from sklearn.preprocessing import StandardScaler

baseDir = 'data/california/california/train'

# meta_data = pd.read_csv(os.path.join(baseDir,'BG_METADATA_2016.csv'))
counts_data = pd.read_csv(os.path.join(baseDir, 'X19_INCOME.csv'))
ca_sf = shapefile.Reader('data/tl_2018_06_tract')

data = []

def plot_california_counties():
    ca_counties_sf = shapefile.Reader('data/tl_2016_06_cousub/tl_2016_06_cousub')
    for i in range(len(ca_counties_sf.shapes())):
        county_shape = ca_counties_sf.shape(i)
        x_points = np.zeros((len(county_shape.points),1))
        y_points = np.zeros((len(county_shape.points),1))
        for j in range(len(county_shape.points)):
            x_points[j] = county_shape.points[j][0]
            y_points[j] = county_shape.points[j][1]
        plt.plot(x_points,y_points,'k')

def plot_california():
    # Plot california
    us_sf = shapefile.Reader('data/tl_2018_us_state')
    california_shape = us_sf.shape(13)
    x_cali = np.zeros((len(california_shape.points), 1))
    y_cali = np.zeros((len(california_shape.points), 1))

    for i in range(len(california_shape.points)):
        x_cali[i] = california_shape.points[i][0]
        y_cali[i] = california_shape.points[i][1]

    plt.scatter(x_cali, y_cali, s=0.2)


if os.path.exists('d_processed.txt'):
    with open('d_processed.txt') as tdf:
        data = ast.literal_eval(tdf.read())

else:
    for i in trange(int(len(ca_sf.shapes()))):
        ca_tract_shape = ca_sf.shape(i)
        GEOID = ca_sf.record(i)[3]
        tract_df = counts_data[counts_data['GEOID'].str.contains(GEOID, na=True)]
        if len(tract_df['B19013e1']) == 0:
            continue
        tract_data = {'coord': (float(ca_sf.record(i)[-1]), float(ca_sf.record(i)[-2])), 'GEOID': GEOID}
        # print(tract_df['B19013e1'])
        income = tract_df['B19013e1'].mean(skipna=True)
        if math.isnan(income):
            continue
        tract_data['income'] = income
        data.append(tract_data)

    with open('d_processed.txt', 'w') as tdf:
        tdf.write(str(data))

print(len(data))
plt.figure(figsize=(6,8))
plot_california_counties()
scaler = StandardScaler()
income = np.array([data[i]['income'] for i in range(len(data))])
income_scale = scaler.fit_transform(income.reshape(-1, 1))
subset = [(data[i]['coord'][0], data[i]['coord'][1], income_scale[i]) for i in range(len(data))]
print(scaler.transform(subset))
km = KMeans(n_clusters=10, ).fit_predict(scaler.transform(subset))

for i in range(10):
    mean_income = np.mean([income[j] for j in range(len(income)) if km[j] == i])
    plt.scatter([subset[j][0] for j in range(len(subset)) if km[j] == i],
                [subset[j][1] for j in range(len(subset)) if km[j] == i],
                label=f"Cluster {i} -  Mean Income:${mean_income:.2f}",
                s=10)
plt.xlim((-120, -116))
plt.ylim((33, 35))
plt.axis('equal')
plt.legend()
plt.show()
# km.iterate(100)
# plot_california()
# for i in range(km.num_clusters):
#     plt.scatter(km.get_x_cluster(i), km.get_y_cluster(i), label=f"Cluster {i}")
# plt.legend()
# plt.show()
#
# # print(race_data['GEOID'])
