# -*- coding: Utf-8 -*-
import numpy as np
#import skimage.io as ima
import skimage.io as ima
import scipy.io as scio
from lire_mnist import *
__author__ = "Clément"




class RBM:
    def __init__(self, taille_entree, taille_sortie):
        self.taille_entree = taille_entree
        self.taille_sortie = taille_sortie
        self.W = np.random.normal(0, 0.1, (taille_entree, taille_sortie))
        #self.W = np.random.uniform(-0.25, 0.25, (taille_entree, taille_sortie))
        self.W = np.zeros((taille_entree, taille_sortie))
        #self.a = np.random.uniform(-0.25, 0.25, (1, taille_entree))
        self.a = np.zeros((1, taille_entree))
        #self.b = np.random.uniform(-0.25, 0.25, (1, taille_sortie))
        self.b = np.zeros((1, taille_sortie))

    def entree_sortie_RBM(self, entrees):
        #print "b shape",  self.b.shape
        coller_b = [self.b for _ in range(int(entrees.shape[0]))]
        rep_b = np.concatenate(coller_b, axis= 1).reshape((int(entrees.shape[0]),int(self.W.shape[1])))
        #print "W", self.W.shape
        #print (np.dot(entrees, self.W)+rep_b).shape
        #print entrees.shape, self.W.shape, rep_b.shape
        somme = np.dot(entrees, self.W)+rep_b
        #print "shape", somme.shape
        return 1./(1+np.exp(-somme))
    def sortie_entree_RBM(self, sorties):
        coller_a = [self.a for _ in range(int(sorties.shape[0]))]
        rep_a = np.concatenate(coller_a, axis= 1).reshape((int(sorties.shape[0]),int(self.W.T.shape[1]) ))
        #rep_a = np.array([self.a[0,:] for _ in range(int(sorties.shape[0]))])
        #print sorties.shape, self.W.T.shape, rep_a.shape
        somme = np.dot(sorties, self.W.T)+rep_a
        return 1./(1+np.exp(-somme))
    def calcul_softmax(self, entrees):
        #entrees = np.clip(entrees, -10, 10)
        #self.W = np.clip(self.W, -100, 100)
        #self.b = np.clip(self.b, -100, 100)
        coller_b = [self.b for _ in range(int(entrees.shape[0]))]
        #print entrees[5,:]
        rep_b = np.concatenate(coller_b, axis= 1).reshape((int(entrees.shape[0]),int(self.W.shape[1])))
        
        A = np.exp(np.dot(entrees, self.W)+rep_b)
        #if A[0,0] == np.nan:
        #    print entrees, self.W, rep_b
        #print "softmax"
        #print entrees.shape, self.W.shape , rep_b.shape
        #print A[:,0], np.sum(A[:,0])
        #print A.shape
        #print np.sum(A, axis=0).shape
        #print np.sum(A, axis=1).shape
        denom = np.array([np.sum(A, axis = 1) for _ in range(A.shape[1])]).T#.reshape(A.shape)
        #print "denom", denom.shape, np.sum(A, axis = 1).shape
        #denom = np.sum(A, axis = 1)
        #print A[0,:]
        #print denom[0]
        #print self.W
        #print denom
        res = np.divide(A,denom)
        #print res[0,]


        #aa=exp(data_in*RBM.w+repmat(RBM.b,size(data_in,1),1));
	#bb=repmat(sum(aa,2),1,size(aa,2));
	#proba=aa./bb;
        


        #res = np.exp(prod) / denom
        
        #print "softmax"
        #print prod
        #print denom
        #print res
        #print res.shape
        return res
    def train_RBM(self, nombre_iteration_descente, epsilon, taille_mini_batch, entrees):
        for l in range(nombre_iteration_descente):
            np.random.shuffle(entrees)
            #print np.shape(entrees), taille_mini_batch
            for batch in range(int(np.shape(entrees)[0])/taille_mini_batch - 1):
                v0 = entrees[batch*taille_mini_batch:(batch+1)*(taille_mini_batch),:]
                #print "v0", v0.shape
                h0 = self.entree_sortie_RBM(v0)
                #print "h0", h0.shape
                echantillon_h0 = (np.random.uniform(0, 1, (taille_mini_batch, self.taille_sortie)) < h0).astype("int")
                v1 = self.sortie_entree_RBM(echantillon_h0)
                #print "v1", v1.shape
                echantillon_v1 = (np.random.uniform(0, 1 , (taille_mini_batch, self.taille_entree)) < v1).astype("int")
                h1 = self.entree_sortie_RBM(echantillon_v1)
                pos = np.dot(h0.T, v0)
                neg = np.dot(h1.T, echantillon_v1) #ah bon?
                dW = (pos - neg).T
                #print "dW", dW.shape, "W", self.W.shape
                da = np.sum(v0 - echantillon_v1, axis = 0)
                #print "da shape", da.shape, "a", self.a.shape
                db = np.sum(h0 - h1, axis = 0)
                #print "db", db.shape, "b shape", self.b.shape
                self.W += epsilon*dW
                self.a += epsilon*da
                self.b += epsilon*db
            #V = entrees
            #H = self.entree_sortie_RBM(V)
            #Vr = self.sortie_entree_RBM(H)
            #print np.sum((V-Vr)**2)
    def generer_image_RBM(self, nombre_iterations_Gibbs, nombre_image):
        h = np.random.uniform(0, 1, (1, self.taille_sortie))
        liste_image = []
        for _ in range(nombre_image):
            v = np.random.uniform(0, 1, (1, self.taille_entree))
            for _ in range(nombre_iterations_Gibbs):
                echantillon_v = (np.random.uniform(0, 1 , (1, self.taille_entree)) < v).astype("int")
                #np.where(np.random.uniform(0, 1 , (1, self.taille_entree)) > v, 0, 1 )
                h = self.entree_sortie_RBM(echantillon_v)
                echantillon_h = (np.random.uniform(0, 1, (1, self.taille_sortie)) < h).astype("int")
                #np.where(np.random.uniform(0, 1 , (1, self.taille_sortie))  h, 0, 1 )
		        #print "erreur : "+str(np.sum(echantillon_v - donnees))
                v = self.sortie_entree_RBM(echantillon_h)
            liste_image.append(echantillon_v)
        return liste_image






class DBN:
    def __init__(self, nombre_neurones_couches):
        """

        :param nombre_neurones_couches: c'est une liste dont le ième élément représente le nombre
        de neurones dans la couche i

        :return:
        """
        self.nombre_neurones_couches = nombre_neurones_couches
        self.nombre_couches = len(nombre_neurones_couches)
        self.couches = []
        #liste dont le ième élément est un RBM
        self.initialiser_couches()
    def initialiser_couches(self):
        for i in range(len(self.nombre_neurones_couches)-1):
            self.couches.append(RBM(self.nombre_neurones_couches[i], self.nombre_neurones_couches[i+1]))
        #print self.couches
    def entree_sortie_reseau(self, donnees):
        l = []
        res = donnees
        for couche in self.couches[:len(self.couches)-1]:
            res = couche.entree_sortie_RBM(res)
            l.append(res)
        #print res.shape
        l.append(self.couches[len(self.couches) - 1].calcul_softmax(res))
        return l
    def train_DBN(self, nombre_iteration_descente, epsilon, taille_mini_batch, donnees):
        res = donnees
        print("train DBN")
        #print len(self.couches)

        for i in range(len(self.couches)-1):
            print(i)
            self.couches[i].train_RBM(nombre_iteration_descente, epsilon, taille_mini_batch, res)
            res = self.couches[i].entree_sortie_RBM(res)
            #print self.couches[i].W
    def generer_image_DBN(self, nombre_iterations_Gibbs, nombre_image):
        print(len(self.couches)-2)
        tirage = self.couches[len(self.couches)-2].generer_image_RBM(nombre_iterations_Gibbs, nombre_image)
        tirage = tirage[0]
        for i in range(1,len(self.couches)-1):
            indice_couche =  len(self.couches)-2 - i
            proba  = self.couches[indice_couche].sortie_entree_RBM(tirage)
            #tirage = np.where( np.random.uniform(0, 1 ,proba.shape) < proba, 1, 0)
            tirage = (np.random.uniform(0, 1, proba.shape) < proba).astype("int")
        return tirage
    def retropropagation(self, nb_iteration, epsilon, taille_mini_batch, donnees, labels):
        print("début rétropropagation")
        print("labels", labels.shape, "donnees", donnees.shape)
        for _ in range(nb_iteration):
            np.random.shuffle(donnees)
            print(np.shape(donnees))
            for batch in range(int(int(np.shape(donnees)[0])/taille_mini_batch) - 1):
                #print a, k, int(np.shape(donnees)[0])/taille_mini_batch - 1
                entrees = donnees[batch*taille_mini_batch:(batch+1)*(taille_mini_batch),:]
                labels_batch = labels[batch*taille_mini_batch:(batch+1)*(taille_mini_batch),:]
                l_Y = self.entree_sortie_reseau(entrees)
                #print np.sum(np.sum(entrees, axis=1), axis=0), batch*taille_mini_batch, (batch+1)*(taille_mini_batch)
                #print entrees.shape
                #print entrees[:10,:]
                for i in range(len(self.couches)):
                    indice_couche = len(self.couches) - 1 - i
                    Y = l_Y[indice_couche]
                    derivee = Y*(1.0-Y)
                    if indice_couche > 0:
                        X = l_Y[indice_couche - 1]
                    else:
                        X = entrees
                    if indice_couche == len(self.couches) - 1:
                        delta = Y - labels_batch
                        #print "Y : ",Y.shape, " labels : ", labels[batch*taille_mini_batch:(batch+1)*(taille_mini_batch),:].shape
                    else:
                        #print derivee.shape, delta.shape, self.couches[indice_couche+1].W.T.shape
                        delta = derivee * np.dot(delta, self.couches[indice_couche+1].W.T)
                    #print delta.T.shape, X.shape
                    cont = np.concatenate([np.ones((taille_mini_batch, 1)), X], axis = 1)
                    #print cont.T.shape, delta.shape, X.shape
                    machin = np.dot(cont.T, delta)
                    delta_W = machin[1:,:]
                    #print delta_W.shape, self.couches[indice_couche].W.shape
                    delta_b = machin[0,:]
                    #print delta_b.shape, self.couches[indice_couche].b.shape
                    
                    self.couches[indice_couche].W = self.couches[indice_couche].W - epsilon * delta_W/taille_mini_batch
                    self.couches[indice_couche].b = self.couches[indice_couche].b - epsilon * delta_b/taille_mini_batch
                    #print self.couches[indice_couche].W[0,:]
            out = self.entree_sortie_reseau(donnees)
            derniere_couche = self.nombre_couches - 2
            #print out[derniere_couche].shape
            #print labels[:5,:]
            #print out[derniere_couche]
            diff = labels - out[derniere_couche]
            print(np.sum(diff*diff, axis=1).shape)
            print(np.sum(diff*diff, axis=1)[0])
            print("erreur",np.sqrt(np.sum(np.sum(diff*diff, axis=1), axis=0)/diff.shape[0]))
            #print "diff", diff.shape, diff
            #print "taille erreur", diff.shape, labels.shape, np.argmax(out[-1], axis=1).shape
            #print "erreur", np.dot(diff.T, diff)#/diff.shape[1]
    def test_DNN(self, donnees, labels):
        print("donnees shape", donnees.shape)
        estimation = self.entree_sortie_reseau(donnees)
        derniere_couche = self.nombre_couches - 2
        diff = labels - estimation[derniere_couche]
        print(np.sum(diff*diff, axis=1).shape)
        print("erreur", np.sqrt(np.sum(np.sum(diff*diff, axis=1), axis=0)/diff.shape[0]))



def creer_image(matrice):
     ima.imshow(matrice)
     ima.show()
	

def lire_alpha_digit_mat(nom_fichier):
    return scio.loadmat(nom_fichier)['azer']

def lire_alpha_digit(nom_fichier):
    return np.genfromtxt(nom_fichier, delimiter = "\t")

if __name__ == "__main__":

    nombre_iteration_descente = 1500
    epsilon = 0.1
    taille_mini_batch = 13
    nombre_iterations_Gibbs = 2000
    nombre_image = 1
    retropropagation = True
    if not retropropagation :

        #sans rétropopagation
        #nom_fichier = "donnees.txt"
        labels, donnees = read(1)
        indice = 10
        nombre = 39
        #donnees = lire_alpha_digit(nom_fichier)[indice*nombre:(indice+1)*nombre,:]
        #donnees.reshape((39, 20*16))
        nom_fichier_mat = "donnees.mat"
        donnees = lire_alpha_digit_mat(nom_fichier_mat)[indice*nombre:(indice+1)*nombre,:]
        donnees.reshape((39, 20*16))
        #creer_image(donnees[1,:].reshape((16,20)).T) #c'est bien comme ça qu'il faut afficher les images
        #print donnees
        print(donnees.shape)
    
        reseau = DBN([int(donnees.shape[1]), 20, 20])  #, 300, 300, 300])
	
        """
        reseau.train_DBN(nombre_iteration_descente=nombre_iteration_descente, epsilon=epsilon,\
                     taille_mini_batch=taille_mini_batch, donnees=donnees)
        im = reseau.generer_image_DBN(nombre_iterations_Gibbs=nombre_iterations_Gibbs, nombre_image=nombre_image)
        print im.shape
        creer_image(im.reshape((16,20)).T)
        for im in reseau.couches[0].generer_image_RBM(nombre_iterations_Gibbs, 1 ):
            creer_image(im.reshape((16,20)).T)
        """
    else:
        labels, donnees = read(1, "training")
        donnees = donnees.reshape((60000,28*28))
        labels = labels.reshape((60000,1))
        print("labels : ", labels.shape, "données : ", donnees.shape)
        #labels = labels[indice]
        #print labels[:10]
        #donnees = donnees[indice*nombre:(indice+1)*nombre,:]
        print("retropropagation")
        reseau = DBN([int(donnees.shape[1]), 20,20,10])
        reseau.retropropagation(nombre_iteration_descente, epsilon, taille_mini_batch, donnees, labels)
        labels, donnees = read(1, "testing")
        reseau.test_DNN(donnees, labels)



