import pandas as pd
import shapefile
from matplotlib import pyplot as plt


def city_labels():
    P = []

    def get_population():
        xcoord = []
        ycoord = []
        population_num = []
        city_name = []
        big_cities = pd.read_csv("..\data\Cali_cities.csv", sep='\t', delimiter=';')
        df = pd.DataFrame(big_cities)
        for i in df.columns:
            if i == 'population':
                population_num.append(df[i].values)
                print(population_num)
            elif i == 'lat':
                ycoord.append(df[i].values)
                print(ycoord)
            elif i == 'lng':
                xcoord.append(df[i].values)
                print(xcoord)
            elif i == 'city':
                city_name.append(df[i].values)
        return (population_num, xcoord, ycoord, city_name)

    P = get_population()
    for i, city in enumerate(P[3][0]):
        x = P[1][0][i]
        y = P[2][0][i]
        plt.scatter(x, y)
        plt.text(x + 0.1, y + 0.11, city, fontsize=9)

