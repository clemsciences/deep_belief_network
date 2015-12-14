# -*- coding: Utf-8 -*-
import numpy as np
__author__ = "Clément"




class RBM:
    def __init__(self, taille_entree, taille_sortie):
        self.taille_entree = taille_entree
        self.taille_sortie = taille_sortie
        self.W = np.random.uniform(-1, 1, (taille_entree, taille_sortie))
        #self.W = np.ones((taille_entree, taille_sortie))
        self.a = np.random.uniform(-1, 1, (1, taille_entree))
        #self.a = np.ones((1, taille_entree))
        self.b = np.random.uniform(-1, 1, (1, taille_sortie))
        #self.b = np.ones((1, taille_sortie))

    def entree_sortie_RBM(self, entrees):
        #print "b shape",  self.b.shape
        coller_b = [self.b for _ in range(int(entrees.shape[0]))]
        rep_b = np.concatenate(coller_b, axis= 1).reshape((int(entrees.shape[0]),int(self.W.shape[1]) ))
        #print "W", self.W.shape
        #print (np.dot(entrees, self.W)+rep_b).shape
        somme = np.dot(entrees, self.W)+rep_b
        #print "shape", somme.shape
        return 1./(1+np.exp(-somme))
    def sortie_entree_RBM(self, sorties):
        coller_a = [self.a for _ in range(int(sorties.shape[0]))]
        rep_a = np.concatenate(coller_a, axis= 1).reshape((int(sorties.shape[0]),int(self.W.T.shape[1]) ))
        #rep_a = np.array([self.a[0,:] for _ in range(int(sorties.shape[0]))])
        somme = np.dot(sorties, self.W.T)+rep_a
        return 1./(1+np.exp(-somme))
    def calcul_softmax(self, entrees):
        coller_b = [self.b for _ in range(int(entrees.shape[0]))]
        rep_b = np.concatenate(coller_b, axis= 1)
        prod = np.dot(entrees, self.W)+rep_b
        somme = 1./(1+np.exp(-prod))
        return prod / np.sum((1+np.exp(somme)), axis = 1)
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
                neg = np.dot(h1.T, echantillon_v1)
                dW = (pos - neg).T
                #print "dW", dW.shape, "W", self.W.shape
                da = np.sum(v0 - echantillon_v1, axis = 0)
                #print "da shape", da.shape, "a", self.a.shape
                db = np.sum(h0 - h1, axis = 0)
                #print "db", db.shape, "b shape", self.b.shape
                self.W += epsilon*dW
                self.a += epsilon*da
                self.b += epsilon*db
    def generer_image_RBM(self, nombre_iterations_Gibbs, nombre_image):
        h = np.random.uniform(0, 1, (1, self.taille_sortie))
        for _ in range(nombre_iterations_Gibbs):
            echantillon_h = np.where(np.random.uniform(0, 1 , (1, self.taille_sortie)) < h, 0, 1 )
            v = self.sortie_entree_RBM(echantillon_h)
            echantillon_v = np.where(np.random.uniform(0, 1 , (1, self.taille_entree)) < v, 0, 1 )
            h = self.entree_sortie_RBM(echantillon_v)
        return echantillon_v






class DBN:
    def __init__(self, nombre_neurones_couches):
        """

        :param nombre_neurones_couches: c'est une liste dont le ième élément représente le nombre
        de neurones dans la couche i

        :return:
        """
        self.nombre_neurones_couches = nombre_neurones_couches
        self.couches = []
        #liste dont le ième élément est un RBM
        self.initialiser_couches()
    def initialiser_couches(self):
        for i in range(len(self.nombre_neurones_couches)-1):
            self.couches.append(RBM(self.nombre_neurones_couches[i], self.nombre_neurones_couches[i+1]))
        #print self.couches
    def entree_sortie_reseau(self, donnees):
        res = donnees
        for couche in self.couches[:len(self.couches)-1]:
            res = couche.entree_sortie_RBM(res)
        return self.couches[len(self.couches) - 1].calcul_softmax(res)
    def train_DBN(self, nombre_iteration_descente, epsilon, taille_mini_batch, donnees):
        res = donnees
        print "train DBN"
        #print len(self.couches)
        for i in range(len(self.couches)-2):
            self.couches[i].train_RBM(nombre_iteration_descente, epsilon, taille_mini_batch, res)
            res = self.couches[i].entree_sortie_RBM(res)
            #print self.couches[i].W
    def generer_image_DBN(self, nombre_iterations_Gibbs, nombre_image):
        tirage = self.couches[len(self.couches)-1].generer_image_RBM(nombre_iterations_Gibbs, nombre_image)
        for i in range(len(self.couches)-1):
            proba  = self.couches[len(self.couches)-1 - i].entree_sortie_RBM(tirage)
            tirage = np.where( np.random.uniform(0, 1 ,proba.shape) < proba, 0, 1)
        return tirage





def lire_alpha_digit(nom_fichier):
    return np.genfromtxt(nom_fichier, delimiter = "\t")

if __name__ == "__main__":
    nom_fichier = "donnees.txt"
    donnees = lire_alpha_digit(nom_fichier)
    print donnees
    print donnees.shape
    nombre_iteration_descente = 500
    epsilon = 0.1
    taille_mini_batch = 13
    nombre_iterations_Gibbs = 500
    nombre_image = 1
    reseau = DBN([int(donnees.shape[1]), 40, 10, 5, 3])
    reseau.train_DBN(nombre_iteration_descente=nombre_iteration_descente, epsilon=epsilon,\
                     taille_mini_batch=taille_mini_batch, donnees=donnees)
    #reseau.generer_image_DBN(nombre_iterations_Gibbs=nombre_iterations_Gibbs, nombre_image=nombre_image)
    print reseau.couches[0].generer_image_RBM(nombre_iterations_Gibbs, 1)
