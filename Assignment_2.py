import re
import subprocess
import numpy as np
import matplotlib.image as mpimg
import os
import socket
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from glob import iglob
import pandas as pd
from sklearn.cluster import KMeans
#from sklearn.mixture import GMM
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
#import face_recognition

#Insert all the faces
infile_all_data= 'C:\\Users\\Χρήστης\\Desktop\\Μεταπτυχιακό\\Machine Learning\\Assignment 2\\att_faces\\all_data'
#print(infile_all_data)
#Insert your training set path
infile_training= 'C:\\Users\\Χρήστης\\Desktop\\Μεταπτυχιακό\\Machine Learning\\Assignment 2\\att_faces\\training\\'
#Insert your testing set path
infile_testing = 'C:\\Users\\Χρήστης\\Desktop\\Μεταπτυχιακό\\Machine Learning\\Assignment 2\\att_faces\\testing\\'


#Ylopoihsh Askhshs a()


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.
    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
    with open(filename, 'rb') as f:
        buffer_ = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer_).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer_,
                         dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                         count=int(width)*int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))


def path_all_data(path_all_data):
    list_of_images = []
    files = []
    for r,d,f in os.walk(path_all_data):
        for file in f:
            if '.pgm' in file:
                files.append(os.path.join(r,file))
    for f in files:
        list_of_images.append(read_pgm(f))
    list_of_images = np.asarray(list_of_images)
    return list_of_images
##
##
##def path_training(path_training):
##    list_of_images = []
##    files = []
##    for r,d,f in os.walk(path_training):
##        for file in f:
##            if '.pgm' in file:
##                files.append(os.path.join(r,file))
##    for f in files:
##        list_of_images.append(read_pgm(f))
##    list_of_images = np.asarray(list_of_images)
##    return list_of_images
##
##
##def path_testing(path_testing):
##    list_of_images = []
##    files = []
##    for r,d,f in os.walk(path_testing):
##        for file in f:
##            if '.pgm' in file:
##                files.append(os.path.join(r,file))
##    for f in files:
##        list_of_images.append(read_pgm(f))
##    list_of_images = np.asarray(list_of_images)
##    return list_of_images
##
##
def pca_face(faces, i):
    nsamples, nx, ny = faces.shape
    faces = faces.reshape((nsamples, nx*ny))
    faces_std = StandardScaler().fit_transform(faces)
    #create covariance matrix
    faces_pca = PCA(n_components = i, svd_solver = "full")
    #calculate eigenvalues
    principalcomponents = faces_pca.fit_transform(faces_std)
    centered_matrix = faces_std - faces_std.mean(axis=1)[:, np.newaxis]
    cov = np.dot(centered_matrix, centered_matrix.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    pca_components = pd.DataFrame(principalcomponents)
    return eigvals
##
##x = [10, 25, 50, 75, 100]
##for i in x:
##    eigenvals_training = pca_face(path_training(infile_training), i)
##for j in eigenvals_training:
##    eigenvals_testing = pca_face(path_testing(infile_testing), j)
##    
##
##def knn_classifier_euclidean_distance():
##    images = path_training(infile_training)
##    for i in images:
##        face_bounding_boxes = face_recognition.face_locations(i)
##        face_encodings = face_recognition.face_encodings(image, known_face_locations = face_bounding_boxes)
##        classifier = train("image", model_save_path="trained_knn_model.clf", n_neighbors=2, metric="euclidean")
##        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2)
##        are_matches = [min(closest_distances[0][i][0], closest_distances[0][i][1]) <= distance_threshold for i in range(len(face_bounding_boxes))]
##        predictions = predict(infile_testing, model_path="trained_knn_model.clf")
##        show_prediction_label_on_image(count, os.path.join("knn_examples/test", image_file), predictions)
##        
##
##def knn_classifier_cosine_distance():
##    images = path_training(infile_training)
##    for i in images:
##        face_bounding_boxes = face_recognition.face_locations(i)
##        face_encodings = face_recognition.face_encodings(image, known_face_locations = face_bounding_boxes)
##        classifier = train("image", model_save_path="trained_knn_model.clf", n_neighbors=2)
##        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2, metric='cosine')
##        are_matches = [min(closest_distances[0][i][0], closest_distances[0][i][1]) <= distance_threshold for i in range(len(face_bounding_boxes))]
##        predictions = predict(infile_testing, model_path="trained_knn_model.clf")
##        show_prediction_label_on_image(count, os.path.join("knn_examples/test", image_file), predictions)
##    
##
##


#Ylopoihsh Askhshs (b)


def pca_face_1(faces, i):
    nsamples, nx, ny = faces.shape
    faces = faces.reshape((nsamples, nx*ny))
    faces_std = StandardScaler().fit_transform(faces)
    #create covariance matrix
    faces_pca = PCA(n_components = i, svd_solver = "full")
    #calculate eigenvalues
    principalcomponents = faces_pca.fit_transform(faces_std)
    centered_matrix = faces_std - faces_std.mean(axis=1)[:, np.newaxis]
    cov = np.dot(centered_matrix, centered_matrix.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    pca_components = pd.DataFrame(principalcomponents)
    return pca_components

x = [10, 25, 50, 75, 100]
for i in x:
    pca_componets_all_data = pca_face_1(path_all_data(infile_all_data), i)


def k_means_pca_components(pca_components):
    #Sum of squared distances of samples to their closest cluster center
    inertias = []
    #Labels of eacch point
    label_point = []
    #coordinates of cluster centers
    coordinates_of_cluster_centers = []
    #Number of iterations run
    iterations_run = []
    model = KMeans(n_clusters=10)
    model.fit(pca_components.iloc[:,:3])
    inertias.append(model.inertia_)
    label_point.append(model.labels_)
    coordinates_of_cluster_centers.append(model.cluster_centers_)
    iterations_run.append(model.n_iter_)
    return  coordinates_of_cluster_centers


#print(k_means_pca_components(pca_componets_all_data))


##
##def gmm_pca_components(pca_components):
##    gmm = GMM(n_components=10).fit(pca_components)
##    labels = gmm.predict(pca_components)
##    
##
###Compute each cluster with purity distance
##    """Purity score
##
##    To compute purity, each cluster is assigned to the class which is most frequent 
##    in the cluster [1], and then the accuracy of this assignment is measured by counting 
##    the number of correctly assigned documents and dividing by the number of documents.
##    We suppose here that the ground truth labels are integers, the same with the predicted clusters i.e
##    the clusters index.
##
##    Args:
##        y_true(np.ndarray): n*1 matrix Ground truth labels
##        y_pred(np.ndarray): n*1 matrix Predicted clusters
##    
##    Returns:
##        float: Purity score
##    
##    References:
##        [1] https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
##    """
##
##def purity_score_1(y_true, y_pred):
##    """Purity score
##
##    To compute purity, each cluster is assigned to the class which is most frequent 
##    in the cluster [1], and then the accuracy of this assignment is measured by counting 
##    the number of correctly assigned documents and dividing by the number of documents.
##    We suppose here that the ground truth labels are integers, the same with the predicted clusters i.e
##    the clusters index.
##
##    Args:
##        y_true(np.ndarray): n*1 matrix Ground truth labels
##        y_pred(np.ndarray): n*1 matrix Predicted clusters
##    
##    Returns:
##        float: Purity score
##    
##    References:
##        [1] https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
##    """
##    # matrix which will hold the majority-voted labels
##    y_voted_labels = np.zeros(y_true.shape)
##    # Ordering labels
##    ## Labels might be missing e.g with set like 0,2 where 1 is missing
##    ## First find the unique labels, then map the labels to an ordered set
##    ## 0,2 should become 0,1
##    labels = np.unique(y_true)
##    ordered_labels = np.arange(labels.shape[0])
##    for k in range(labels.shape[0]):
##        y_true[y_true==labels[k]] = ordered_labels[k]
##    # Update unique labels
##    labels = np.unique(y_true)
##    # We set the number of bins to be n_classes+2 so that 
##    # we count the actual occurence of classes between two consecutive bin
##    # the bigger being excluded [bin_i, bin_i+1[
##    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)
##
##    for cluster in np.unique(y_pred):
##        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
##        # Find the most present label in the cluster
##        winner = np.argmax(hist)
##        y_voted_labels[y_pred==cluster] = winner
##    
##    return accuracy_score(y_true, y_voted_labels)
##
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

print(purity_score(k_means_pca_components(pca_componets_all_data),pca_componets_all_data ))
