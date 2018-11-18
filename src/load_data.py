import os, ast
import numpy as np
import pandas as pd
from tqdm import trange
import math
import shapefile
from matplotlib import pyplot as plt
from sklearn.cluster.k_means_ import KMeans
from sklearn.preprocessing import StandardScaler
from route_algo import min_span_tree
from Cali_map import city_labels

baseDir = '../data/california/california/train'

# meta_data = pd.read_csv(os.path.join(baseDir,'BG_METADATA_2016.csv'))

# counts_data = pd.read_csv(os.path.join(baseDir, 'X19_INCOME.csv'))
counts_data = pd.read_csv(os.path.join(baseDir, 'X19_INCOME.csv'))
commuting_data = pd.read_csv(os.path.join(baseDir, 'X08_COMMUTING.csv'))
ca_tract_sf = shapefile.Reader('../data/tl_2018_06_bg')


def plot_california_counties():
    ca_counties_sf = shapefile.Reader('../data/tl_2018_us_county')
    for i in range(len(ca_counties_sf.shapes())):
        if ca_counties_sf.record(i)[0] != '06': continue
        county_shape = ca_counties_sf.shape(i)
        x_points = np.zeros((len(county_shape.points), 1))
        y_points = np.zeros((len(county_shape.points), 1))
        for j in range(len(county_shape.points)):
            x_points[j] = county_shape.points[j][0]
            y_points[j] = county_shape.points[j][1]
        plt.plot(x_points, y_points, 'k', alpha=0.05)


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


def process_county_data(ca_sf, df, col_label, col_label_verbose, path='dc_processed.txt', intrinsic=True):
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
    if os.path.exists(path):
        with open(path) as pcdpf:
            data = ast.literal_eval(pcdpf.read())
            return data

    if not intrinsic:
        county_pop = {}
        for c_G_ID in df['GEOID']:
            county_df = population_data[population_data['GEOID'].str.contains(c_G_ID, na=True)]

            county_pop[c_G_ID] = county_df['B00001e1'].sum(skipna=True)
    for tract_id in trange(len(ca_sf.shapes())):
        GEOID = ca_sf.record(tract_id)[4]
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
    with open(path, 'w') as pcdpf:
        pcdpf.write(str(data))
    return data


def process_data(ca_sf, col_label, col_label_verbose, df, path='d_processed.txt', fill=None):
    data = [] if not fill else fill
    if os.path.exists(path):
        with open(path) as tdf:
            data = ast.literal_eval(tdf.read())
            return data
    else:
        for i in trange(int(len(ca_sf.shapes()))):
            GEOID = ca_sf.record(i)[4]
            tract_df = df[df['GEOID'] == f"15000US{GEOID}"]
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


def sum_cluster(cluster_ids, cluster, data):
    if len(data) != len(cluster_ids):
        raise ValueError("data and cluster_ids must have identical lengths")
    return sum(data[i] for i in cluster_ids if i == cluster)


jobs_employment = pd.read_csv('../data/Jobs_employment_2.csv', delimiter=';')
jobs_data = process_county_data(ca_tract_sf, 'Number of Jobs', 'Number of Jobs', jobs_employment, path='jobs_data.txt',
                                intrinsic=False)
emp_end_df = pd.read_csv('../data/Employment_Education_status.csv')
empl_data = process_data(ca_tract_sf, 'SIGNAL', 'Employment/Education Status', emp_end_df, 'emp_end.txt')

plt.figure(figsize=(6, 8))
plot_california_counties()
scaler = StandardScaler()
n_o_j = np.array([jobs_data[i]['Number of Jobs'] for i in range(len(jobs_data))])
n_o_j_scale = scaler.fit_transform(n_o_j.reshape(-1, 1))
subset = [(jobs_data[i]['coord'][0], jobs_data[i]['coord'][1], n_o_j_scale[i]) for i in range(len(jobs_data))]
emp_edu = np.array([empl_data[i]['Employment/Education Status'] for i in range(len(empl_data))])
empl_data_scale = scaler.fit_transform(emp_edu.reshape(-1, 1))
subset_empl_edu = [(empl_data[i]['coord'][0], empl_data[i]['coord'][1], empl_data_scale[i][0]) for i in
                   range(len(empl_data))]


def plot_network(n_clusters, subset_job, subset_edu, no_jobs, no_edu):
    plt.figure(figsize=(6, 8))
    job_kmeans = KMeans(n_clusters=n_clusters)
    job_predict = job_kmeans.fit_predict(subset_job)
    empl_edu_kmean = KMeans(n_clusters=n_clusters)
    empl_predict = empl_edu_kmean.fit_predict(subset_edu)

    cluster_sum_jobs, cluster_sum_employ_edu = [], []

    for i in range(n_clusters):
        cluster_sum_employ_edu.append(sum_cluster(empl_predict, i, no_edu) / sum(no_edu))
        cluster_sum_jobs.append(sum_cluster(job_predict, i, no_jobs) / sum(no_jobs))

    jobs_centres = job_kmeans.cluster_centers_
    emp_edu_centres = empl_edu_kmean.cluster_centers_
    result, all_coords = min_span_tree(jobs_centres, emp_edu_centres, cluster_sum_jobs, cluster_sum_employ_edu)
    city_labels()
    plot_california_counties()
    plot_california()
    for i in range(len(result)):
        for j in range(len(result[i])):
            if result[i][j] == 0:  # NO LINK
                continue

            plt.scatter(jobs_centres[i][0] if i < n_clusters else emp_edu_centres[i - n_clusters][0],
                        jobs_centres[i][1] if i < n_clusters else emp_edu_centres[i - n_clusters][1], edgecolors='b',
                        facecolors='none')
            plt.scatter(jobs_centres[j][0] if j < n_clusters else emp_edu_centres[j - n_clusters][0],
                        jobs_centres[j][1] if j < n_clusters else emp_edu_centres[j - n_clusters][1], edgecolors='b',
                        facecolor='none')
            plt.plot((jobs_centres[i][0] if i < n_clusters else emp_edu_centres[i - n_clusters][0],
                      jobs_centres[j][0] if j < n_clusters else emp_edu_centres[j - n_clusters][0]),
                     (jobs_centres[i][1] if i < n_clusters else emp_edu_centres[i - n_clusters][1],
                      jobs_centres[j][1] if j < n_clusters else emp_edu_centres[j - n_clusters][1]), 'b-')

    plt.show()


def plot_job_cluster(n_clusters, no_jobs, subset, kmeans=None):
    if not kmeans:
        kmeans = KMeans(n_clusters=n_clusters)
    job_predict = kmeans.fit_predict(subset)
    plot_california_counties()
    for i in range(n_clusters):
        mean_jobs = np.mean([no_jobs[j] for j in range(len(no_jobs)) if job_predict[j] == i])
        plt.scatter([subset[j][0] for j in range(len(subset)) if job_predict[j] == i],
                    [subset[j][1] for j in range(len(subset)) if job_predict[j] == i],
                    label=f"Mean No. Jobs:{mean_jobs:.0f}",
                    s=4.5)

    # city_labels()
    plt.legend()
    plt.gca().set_xlabel("Longitude")
    plt.gca().set_ylabel("Latitude")
    plt.xlim((-120, -116))
    plt.ylim((33, 35))
    plt.axis('equal')
    plt.show()


def plot_employment_edu_cluster(n_clusters, no_edu, subset_edu, kmeans=None):
    if not kmeans:
        kmeans = KMeans(n_clusters=n_clusters)
    empl_predict = kmeans.fit_predict(subset_edu)
    plot_california()
    plot_california_counties()
    for i in range(n_clusters):
        mean_employment_score = np.mean([no_edu[j] for j in range(len(no_edu)) if empl_predict[j] == i])
        plt.scatter([subset_edu[j][0] for j in range(len(subset_edu)) if empl_predict[j] == i],
                    [subset_edu[j][1] for j in range(len(subset_edu)) if empl_predict[j] == i],
                    label=f"Mean Employment Score:{mean_employment_score:.5f}",
                    s=4.5)
    plt.legend()
    plt.gca().set_xlabel("Longitude")
    plt.gca().set_ylabel("Latitude")
    plt.xlim((-120, -116))
    plt.ylim((33, 35))
    plt.axis('equal')
    plt.show()

for i in np.arange(5,100,5):
    plot_network(int(i),subset,subset_empl_edu,n_o_j,emp_edu)
