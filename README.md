# NYUC-CS-MachineLearning

## HW1.
Implement rLSE,Newton's methon to find the best linear model fitting data points

## HW2. 
HandCraft Naive Bayes classifier
Create a Naive Bayes classifier for each handwritten digit that support discrete and continuous
features.



## HW3.
Baysian Linear regression
Online learning method for implementing predictive distribution

### Note:
1. Discrete Version:

Tally the frequency of the values of each pixel into 32 bins. 
For example, The gray level 0 to 7 should be classified to bin 0, gray level 8 to 15 should be bin 1 

2. Continuous Version:
Use MLE to fit a Gaussian distribution for the value of each pixel(0-256). Perform Naive
Bayes classifier.

## HW4.
1. Logistic Regression using gradient descent, newton's method
2. EM algorithm for unsupervised learning (MINIST classification)

## HW5.
1. Gaussian Process

### Before Optimization:
Predict the distribution of data ranging from -60 to 60, where the GP model is trained by 34 points.

### With Optimization:
Apply some initials into kernel, and then plug kernel into log likelihood for finding the best parameters(MLE)

2. SVM

Implement SVM with different kernels to do MINIST binary classification.

## HW6.
Code out kernel k-means, spectral clustering (both normalized cut and ratio cut), considering spatial similarity and color similarity upon the clustering.

Two 100*100 images are provided, and each pixel in the image should be treated as a data point. Grouping the pixels into different groups by colors

## HW7.

### PART1.
1. Use PCA and LDA to show the first 25 eigenfaces and fisherfaces, and randomly pick 10 images to show their reconstruction.
2. Use PCA and LDA to do face recognition, and compute the performance. Use k-nearest neighbor to classify which subject the testing image belongs to.
3. Use kernel PCA and kernel LDA to do face recognition, and compute the performance. (kenels: RBF + Polynomial)

### PART2.
Visualize the embedding of both t-SNE and symmetric SNE. (minist)
