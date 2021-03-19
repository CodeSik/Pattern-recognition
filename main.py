import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from sklearn.cluster import KMeans

def generateSample(m_x,sig_x,m_y,sig_y,m_z,sig_z,class_num):
    X = np.random.normal(m_x, sig_x, 300)
    Y = np.random.normal(m_y, sig_y, 300)
    Z = np.random.normal(m_z, sig_z, 300)

    for i in range(0,300):
        df.loc[class_num+i] = [X[i],Y[i],Z[i]]


def mergeSample(c1,c2,c3,c4,c5):
    Total_data = []
    Total_data.append(c1)
    Total_data.append(c2)
    Total_data.append(c3)
    Total_data.append(c4)
    Total_data.append(c5)
    return Total_data

def drawDistribution(ax):
    for i in range(0, len(class1_vector)):
        ax.scatter(class1_vector[i][0], class1_vector[i][1], class1_vector[i][2], color='blue',marker='o',s=50)
    for i in range(0, len(class2_vector)):
        ax.scatter(class2_vector[i][0], class2_vector[i][1], class2_vector[i][2], color='red')
    for i in range(0, len(class3_vector)):
        ax.scatter(class3_vector[i][0], class3_vector[i][1], class3_vector[i][2], color='green')
    for i in range(0, len(class4_vector)):
        ax.scatter(class4_vector[i][0], class4_vector[i][1], class4_vector[i][2], color='indigo')
    for i in range(0, len(class5_vector)):
        ax.scatter(class5_vector[i][0], class5_vector[i][1], class5_vector[i][2], color='orange')

def performKmean():
    model = KMeans(n_clusters=5, init="k-means++").fit(Total_data)
    model.cluster_centers_

    ax.scatterplot(x="x",y="y",z="z",hue = "cluster",data = result_by_sklearn, palette = "Set2")


if __name__ == '__main__':
    df = pd.DataFrame(columns=['x','y','z'])
    class1_vector = generateSample(0, 1, 0, 1, 1, 2,0)
    class2_vector = generateSample(2, 1, 2, 1, 3, 2,300)
    class3_vector = generateSample(4, 1, 4, 1, 5, 2,600)
    class4_vector = generateSample(6, 1, 6, 1, 7, 2,900)
    class5_vector = generateSample(8, 1, 8, 1, 9, 2,1200)

    df.head(100)
    # style.use('seaborn-talk')
    # mpl.rcParams['legend.fontsize'] = 10
    # fig = plt.figure()
    # ax = fig.gca(projection = '3d')
    # # drawDistribution(ax)
    # # print(compact,centers,res)
    # performKmean()
    # plt.show()

