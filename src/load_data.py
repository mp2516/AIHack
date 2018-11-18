import os, ast
import numpy as np
import pandas as pd
from tqdm import trange
import math
import shapefile
from matplotlib import pyplot as plt
from sklearn.cluster.k_means_ import KMeans
from sklearn.preprocessing import StandardScaler
from .Route_Algo import min_span_tree
baseDir = '../data/california/california/train'

# meta_data = pd.read_csv(os.path.join(baseDir,'BG_METADATA_2016.csv'))

counts_data = pd.read_csv(os.path.join(baseDir, 'X19_INCOME.csv'))
ca_tract_sf = shapefile.Reader('../data/tl_2018_06_tract')


def plot_california_counties():
    ca_counties_sf = shapefile.Reader('../data/tl_2018_us_county')
    for i in range(len(ca_counties_sf.shapes())):
        if (ca_counties_sf.record(i)[0] != '06'): continue
        county_shape = ca_counties_sf.shape(i)
        x_points = np.zeros((len(county_shape.points), 1))
        y_points = np.zeros((len(county_shape.points), 1))
        for j in range(len(county_shape.points)):
            x_points[j] = county_shape.points[j][0]
            y_points[j] = county_shape.points[j][1]
        plt.plot(x_points, y_points, 'k')


def plot_california():
    # Plot california
    us_sf = shapefile.Reader('../data/tl_2018_us_state')
    california_shape = us_sf.shape(13)
    x_cali = np.zeros((len(california_shape.points), 1))
    y_cali = np.zeros((len(california_shape.points), 1))

    for i in range(len(california_shape.points)):
        x_cali[i] = california_shape.points[i][0]
        y_cali[i] = california_shape.points[i][1]

    plt.scatter(x_cali, y_cali, s=0.2)


def process_county_data(ca_sf, col_label, col_label_verbose, df, path='dc_processed.txt', intrinsic=True):
    """
    Function for processing county data - more zoomed out than tract dataset - use interpolation if not intrisinic
    :param ca_sf:
    :param col_label:
    :param col_label_verbos:
    :param df:
    :param path:
    :param intrinsic:
    :return:
    """
    population_data = pd.read_csv(os.path.join(baseDir, 'X00_COUNTS.csv'))
    population_data = population_data[['GEOID', 'B00001e1', 'B00001m1']]
    data = []
    if not intrinsic:
        county_pop = {}
        for c_G_ID in df['GEOID']:
            county_df = population_data[population_data['GEOID'].str.contains(c_G_ID, na=True)]

            county_pop[c_G_ID] = county_df['B00001e1'].sum(skipna=True)
    for tract_id in trange(len(ca_sf.shapes())):
        GEOID = ca_sf.record(tract_id)[3]
        county_GEOID = GEOID[:5]
        tract_df = df[df['GEOID'] == f"15000US{county_GEOID}"]
        tract_data = {'coord': (float(ca_sf.record(tract_id)[-1]), float(ca_sf.record(tract_id)[-2])),
                      'GEOID': GEOID}
        if not intrinsic:
            tract_pop = population_data[population_data['GEOID'].str.contains(GEOID, na=True)]['B00001e1'].sum(
                skipna=True)
            tract_data[col_label_verbose] = float(tract_df[col_label].iloc[0].replace(',', '')) * (
                        tract_pop / county_pop[f"15000US{county_GEOID}"])
        else:
            tract_data[col_label_verbose] = tract_df[col_label].iloc[0]
        data.append(tract_data)
    return data


def process_data(ca_sf, col_label, col_label_verbose, df, path='d_processed.txt'):
    data = []
    if os.path.exists(path):
        with open(path) as tdf:
            data = ast.literal_eval(tdf.read())
            return data
    else:
        for i in trange(int(len(ca_sf.shapes()))):
            GEOID = ca_sf.record(i)[3]
            tract_df = df[df['GEOID'].str.contains(GEOID, na=True)]
            if len(tract_df[col_label]) == 0:
                continue
            tract_data = {'coord': (float(ca_sf.record(i)[-1]), float(ca_sf.record(i)[-2])), 'GEOID': GEOID}
            avg_param = tract_df[col_label].mean(skipna=True)
            if math.isnan(avg_param):
                continue
            tract_data[col_label_verbose] = avg_param
            data.append(tract_data)

        with open(path, 'w') as tdf:
            tdf.write(str(data))
    return data


jobs_employment = pd.read_csv('../jobs_data/Jobs_employment_2.csv', delimiter=';')
data = process_county_data(ca_tract_sf, 'Number of Jobs', 'Number of Jobs', jobs_employment, intrinsic=False)
print(data)
# plot_california_counties()
# plt.scatter([data[i]['coord'][0] for i in range(len(data))], [data[i]['coord'][1] for i in range(len(data))],
#             c=[data[i]['Number of Jobs'] for i in range(len(data))], s=5)
# plt.show()
# data = process_data(ca_tract_sf, 'B19013e1', 'income', counts_data)
plt.figure(figsize=(6, 8))
plot_california_counties()
scaler = StandardScaler()
n_o_j = np.array([data[i]['Number of Jobs'] for i in range(len(data))])
n_o_j_scale = scaler.fit_transform(n_o_j.reshape(-1, 1))
subset = [(data[i]['coord'][0], data[i]['coord'][1], n_o_j_scale[i]) for i in range(len(data))]
print(scaler.transform(subset))
n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters)
km = kmeans.fit_predict(scaler.transform(subset))

for i in range(n_clusters):
    mean_jobs = np.mean([n_o_j[j] for j in range(len(n_o_j)) if km[j] == i])
    plt.scatter([subset[j][0] for j in range(len(subset)) if km[j] == i],
                [subset[j][1] for j in range(len(subset)) if km[j] == i],
                label=f"Mean Income:${mean_jobs:.2f}",
                s=5)


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
