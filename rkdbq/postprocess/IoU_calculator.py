from shapely.geometry import Polygon

a = Polygon([(1, 1), (5, 1), (5, 4), (1, 4)])
b = Polygon([(1, 1), (5, 1), (5, 4), (1, 4)])

print(a.intersection(b).area)
print(a.union(b).area)