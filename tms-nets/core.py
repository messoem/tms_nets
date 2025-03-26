import tms-nets.utils as utils

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

