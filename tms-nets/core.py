import tms-nets.utils as utils
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def get_plot(p):
    if (p.shape)[1] == 2:
        df = pd.DataFrame(p, columns=["x", "y"])
        #print(df)
        sns.scatterplot(df, x="x", y="y").plot()
    elif (p.shape)[1] == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        df = pd.DataFrame(p, columns=["x", "y", "z"])
        ax.scatter(df['x'], df['y'], df['z'])
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.title('3D Scatter Plot with Seaborn')
        plt.show()

def get_points(b, t, m, s):
    G = utils.generate_generator_matrices(b, t, m, s)
    points = np.ones((b**m, s))
    for i in range(s):
        for n in range(0, b**m):
            #print(G[i]@ utils.vecbm(b, m, n))
            tmp = G[i]@ utils.vecbm(b, m, n)
            for j in range(len(tmp)):
                tmp[j] = tmp[j] % b
            points[n][i] = utils.rnum(b, tmp)*b**(-m)
    return points

