import numpy as np
a = open("aux.txt")
f = a.readlines()
numeros = [archivo.split(" ")[-1].strip() for archivo in f]
n = list(map(float,numeros))
print("ACC MV: ",np.mean(n))
