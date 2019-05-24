# a function to match near neighbors to satellites in bins of color and magnitude

def chooseSatellites(index, near, Nsat, ):


zmag0 = zmag[index[0]]
rzcolor0 = rzcolor[index[0]]
index0 = np.array(index[0])

flat_near = []
for i in range(len(edges[1]) - 1):
    for j in range(len(edges[0]) - 1):
        a = index0[np.where((zmag0 >= edges[1][i]) & (zmag0 <= edges[1][i + 1]) & (rzcolor0 >= edges[0][j]) & (
                rzcolor0 <= edges[0][j + 1]))]
        if len(a) > 0:
            flat_near.append(a)
        else:
            flat_near.append(-999)

# b = np.array(b)
# print(type(b))
# print(type(b[1500]))
# print(b[2473])

rz_v_zmag = Nsat[0][:, :, :].sum(axis=2)
# rz_vs_zmag = np.flipud(rz_v_zmag)
flat_Nsat = rz_v_zmag.ravel(order='F')


# print(flat_Nsat[np.where(flat_Nsat > 0.)])
# print(len(flat_Nsat))

# print(np.random.choice(b[2473], size=2, replace=False))

def round_half_up(n, decimals):
    import math

    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier


n = int(round_half_up(flat_Nsat[1725], 0))
print(n)
print(flat_Nsat[1725])
print(flat_near[1725])
print(type(flat_near[1725]))

ind = []
for i in range(len(flat_Nsat)):
    if flat_Nsat[i] > 0.5:
        n = round_half_up(flat_Nsat[i], 0)
        ind.append(np.random.choice(flat_near[i], size=int(n), replace=False))
    else:
        pass
#     print(ind)
#     ind = []
#     print(i)

# a = []
# for i in range(len(ind)):
#     if len(ind[i]) > 0:
#         a.append(ind[i])
#     else:
#         pass

a = np.concatenate(ind)
print(a)
plt.scatter(zmag[a], rzcolor[a])
plt.xlabel(r'$zmag$')
plt.ylabel(r'$(r-z)$ $color$')
plt.gca().invert_xaxis()
plt.show()