import csv, ast
import os
import pandas as pd
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
import shapefile
import math
from sklearn.cluster.k_means_ import KMeans
from sklearn.preprocessing import StandardScaler

file_name = 'C:\\Users\\18072\\PycharmProjects\\AIHack\\misc_data\\county_to_geoid.csv'

county_to_geoid = {}
with open(file_name) as fh:
    for line in fh:
        geo_id, county = line.strip().split(',')
        county_to_geoid[geo_id] = county.strip()

baseDir = 'C:\\Users\\18072\\PycharmProjects\\AIHack'

employment_education = pd.read_csv(os.path.join(baseDir, 'data_employment\\Employment_Education_status.csv'))
ca_sf = shapefile.Reader('C:\\Users\\18072\\PycharmProjects\\AIHack\\map\\tl_2018_06_tract\\tl_2018_06_tract')

data = []

def plot_california_counties():
    ca_counties_sf = shapefile.Reader('C:\\Users\\18072\\PycharmProjects\\AIHack\\map\\tl_2018_06_tract\\tl_2018_06_tract')
    for i in trange(len(ca_counties_sf.shapes())):
        if ca_counties_sf.record(i)[0] != '06': continue
        county_shape = ca_counties_sf.shape(i)
        x_points = np.zeros((len(county_shape.points), 1))
        y_points = np.zeros((len(county_shape.points), 1))
        for j in range(len(county_shape.points)):
            x_points[j] = county_shape.points[j][0]
            y_points[j] = county_shape.points[j][1]
        plt.scatter(x_points, y_points, s=0.1, c='k')


def plot_california():
    # Plot california
    us_sf = shapefile.Reader('C:\\Users\\18072\\PycharmProjects\\AIHack\\map\\tl_2018_us_state\\tl_2018_us_state')
    california_shape = us_sf.shape(13)
    x_cali = np.zeros((len(california_shape.points), 1))
    y_cali = np.zeros((len(california_shape.points), 1))

    for i in trange(len(california_shape.points)):
        x_cali[i] = california_shape.points[i][0]
        y_cali[i] = california_shape.points[i][1]

    plt.scatter(x_cali, y_cali, s=0.2, c='k')

if os.path.exists('d_processed.txt'):
    with open('d_processed.txt') as tdf:
        data = ast.literal_eval(tdf.read())

else:
    for i in trange(int(len(ca_sf.shapes()))):
        ca_tract_shape = ca_sf.shape(i)
        GEOID = ca_sf.record(i)[3]
        tract_df = employment_education[employment_education['GEOID'].str.contains(GEOID, na=True)]
        # if len(tract_df['B19013e1']) == 0:
        #     continue
        tract_data = {'coord': (float(ca_sf.record(i)[-1]), float(ca_sf.record(i)[-2])), 'GEOID': GEOID}
        # print(tract_df['B19013e1'])
        data_employee = tract_df['SIGNAL'].mean(skipna=True)
        if math.isnan(data_employee):
            continue
        tract_data['income'] = data_employee
        data.append(tract_data)

    with open('d_processed.txt', 'w') as tdf:
        tdf.write(str(data))

print(len(data))
plt.figure(figsize=(6, 8))
plot_california()
plot_california_counties()
scaler = StandardScaler()
data_employee = np.array([data[i]['income'] for i in range(len(data))])
income_scale = scaler.fit_transform(data_employee.reshape(-1, 1))
subset = [(data[i]['coord'][0], data[i]['coord'][1], income_scale[i]) for i in range(len(data))]
print(scaler.transform(subset))
km = KMeans(n_clusters=10, ).fit_predict(scaler.transform(subset))

for i in range(10):
    mean_score = np.mean([data_employee[j] for j in range(len(data_employee)) if km[j] == i])
    plt.scatter([subset[j][0] for j in range(len(subset)) if km[j] == i],
                [subset[j][1] for j in range(len(subset)) if km[j] == i],
                label=f"Cluster {i} -  Mean Score:{mean_score:.2f}",
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



