import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import glob



imagenes = glob.glob("imagenes/*.png")
n_imagenes = len(imagenes)
print(np.shape( plt.imread(imagenes[3])))
data = []
for imagen in imagenes:
    data.append(plt.imread(imagen).reshape((-1)))
    #data.append(plt.imread(imagen).reshape((-1,3)))
data = np.array(data)
print(np.shape(data))
inercia = []
centro = []
for i in range(1,20):
    n_clusters = i
    k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
    k_means.fit(data)
    # calculo a cual cluster pertenece cada pixel
    Centros = k_means.cluster_centers_
    centro.append(Centros)
    inercia.append(k_means.inertia_)
    
plt.plot(range(1,20), inercia)
plt.xlim(1,20)
plt.title('Inercia en Funcion de # de clusters')
plt.xlabel('Numero de clusters')
plt.ylabel('Inercia')
plt.grid(b=True)
plt.savefig('inercia.png')

#Podemos Ver que el mejor k es 4
mejor_k = 4
losCentrosqueSon = centro[mejor_k-1]

dis = []
plt.figure(figsize=(16,16))

for i in range(mejor_k):
    distancias = []
    for j in range(n_imagenes):
        d1 = np.linalg.norm(data[j,:] - losCentrosqueSon[i])
        distancias.append(d1)
    distancias = np.array(distancias)
    ii = np.argsort(-distancias)
    distancias = distancias[ii]
    organizadas = data[ii,:]
    LasImagenes = organizadas[0:5,:]
    print(np.shape(LasImagenes))
    for j in range(5):
        data_centered = LasImagenes[j].reshape((100,100,3))
        num_imag = 5*i + j+1
        plt.subplot(mejor_k,5,num_imag)
        plt.imshow(data_centered)
plt.savefig('ejemplo_clases.png')
        
    
    
