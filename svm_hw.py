
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
style.use("ggplot")

path = "/MSAN_USF/courses_spring/630_AdvMachinelearning/code/data/svm-hw/"

def plot_linear_svm(filename, title):

    df = pd.read_csv(path+filename, names=['x1','x2','y'], header=None)
    # df = pd.read_csv(path+"P3_outliar.txt", names=['x1','x2','y'], header=None)

    columns_ls = []
    for column in df.columns:
        columns_ls.append(column)

    X = df[columns_ls[0:len(columns_ls)-1]].values



    Y = df[columns_ls[len(columns_ls)-1]].values
    fig = plt.figure()
    fig.suptitle(title, fontsize=20)

    clf = svm.SVC(kernel='linear')

    clf.fit(X, Y)
    w = clf.coef_[0]
    print "Weights SVM W0=%.2f and W1=%.2f"%(w[0], w[1])
    a = -w[0]/w[1]
    xx =np.linspace(-12, 34)
    yy = a*xx-clf.intercept_[0]/w[1]
    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    # plot the line, the points, and the nearest vectors to the plane
    # plt.plot(xx, yy, 'k-')
    plt.text(0, 10, "Y=+1", fontsize=12)
    plt.text(10, 0, "Y=-1", fontsize=12)
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.plot(xx,yy,color="black", label ="svm decision boundary")

    model = LogisticRegression()
    model.fit(X, Y)
    w = model.coef_[0]
    a = -w[0]/w[1]
    print "Weights Logistic W0=%.2f and W1=%.2f"%(w[0], w[1])
    xx = np.linspace(-12, 34)
    yy = a*xx-model.intercept_[0]/w[1]

    plt.plot(xx,yy, label ="logistic decision boundary")

    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X, Y)
    w = clf_lda.coef_[0]
    a = -w[0]/w[1]
    print "Weights LDA W0=%.2f and W1=%.2f"%(w[0], w[1])
    xx = np.linspace(-12, 34)
    yy = a*xx-clf_lda.intercept_[0]/w[1]
    plt.plot(xx,yy, color="blue", label ="LDA decision boundary")

    # plt.scatter(X[:,0], X[:,1], c=Y)
    # print clf.support_vectors_[:, 0]
    # print clf.support_vectors_[:, 1]

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, color='b')
    plt.scatter(X[:, 0], X[:, 1], c=Y)#, cmap=plt.cm.Paired
    plt.xlabel('X1', fontsize=18)
    plt.ylabel('X2', fontsize=16)
    plt.axis('tight')
    plt.legend()

    plt.show()
    # print clf.predict([.5, .7])

def high_dimensional():
    df_tr = pd.read_csv(path+"P4_train.txt", header=None)

    df_te = pd.read_csv(path+"P4_test.txt", header=None)
    columns_ls = []
    for column in df_tr.columns:
        columns_ls.append(column)

    X_train = df_tr[columns_ls[0:len(columns_ls)-1]].values
    y_train = df_tr[columns_ls[len(columns_ls)-1]].values

    X_test = df_te[columns_ls[0:len(columns_ls)-1]].values
    y_test = df_te[columns_ls[len(columns_ls)-1]].values


    C_range = np.arange(.01, 1., .2)
    gamma_range = np.arange(.1, 1., .2)
    kern = ['poly']
    param_grid = dict(gamma=gamma_range, C=C_range, kernel=kern)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid,cv=None)
    grid.fit(X_train, y_train)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

    y_pred = grid.predict(X_test)
    accScore = accuracy_score(y_test, y_pred)
    print ("Accuracy score for Quadratic Kernel is %.2f"%(accScore))

    kern_rbf = ['rbf']
    param_grid_rbf = dict(gamma=gamma_range, C=C_range, kernel=kern_rbf)
    grid_rbf = GridSearchCV(svm.SVC(), param_grid=param_grid_rbf, cv=5)
    grid_rbf.fit(X_train, y_train)
    #
    print("The best parameters are %s with a score of %0.2f"
          % (grid_rbf.best_params_, grid_rbf.best_score_))

    y_pred_rbf = grid_rbf.predict(X_test)
    accScoreRbf = accuracy_score(y_test, y_pred_rbf)
    print ("Accuracy score for Guassian Kernel is %.2f"%(accScoreRbf))

    k = range(1,10,1)
    parameter = {'n_neighbors': k}
    knn = KNeighborsClassifier()
    knn_clf = GridSearchCV(knn, parameter, cv=5)
    knn_clf.fit(X_train, y_train)
    print("The best parameters for knn %s with a score of %0.2f"
          % (knn_clf.best_params_, knn_clf.best_score_))

    knn_y_pred = knn_clf.predict(X_test)
    knn_accScore = accuracy_score(y_test, knn_y_pred)
    print ("Accuracy score for KNN is %.2f"%(knn_accScore))

def plot_linear_svm_only(filename, title, filename_fig):

    df = pd.read_csv(path+filename, names=['x1','x2','y'], header=None)
    # df = pd.read_csv(path+"P3_outliar.txt", names=['x1','x2','y'], header=None)
    fig = plt.figure()
    fig.suptitle(title, fontsize=20)
    columns_ls = []
    for column in df.columns:
        columns_ls.append(column)

    X = df[columns_ls[0:len(columns_ls)-1]].values
    # X1 =df[columns_ls[0]].values
    #
    #
    # X2 = df[columns_ls[1]].values

    # print np.min(X1), np.max(X1)
    # print np.min(X2), np.max(X2)

    Y = df[columns_ls[len(columns_ls)-1]].values


    clf = svm.SVC(kernel='linear')

    clf.fit(X, Y)
    w = clf.coef_[0]
    print "Weights W0 %.2f and W1%.2f"%(w[0], w[1])
    a = -w[0]/w[1]
    xx =np.linspace(-12, 34)
    yy = a*xx-clf.intercept_[0]/w[1]
    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    # plot the line, the points, and the nearest vectors to the plane
    # plt.plot(xx, yy, 'k-')
    plt.text(0, 10, "Y=+1", fontsize=12)
    plt.text(10, 0, "Y=-1", fontsize=12)
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.plot(xx,yy,color="black", label ="svm decision boundary")


    plt.xlabel('X1', fontsize=18)
    plt.ylabel('X2', fontsize=16)
    # fig.savefig(filename_fig)
    # model = LogisticRegression()
    # model.fit(X, Y)
    # w = model.coef_[0]
    # a = -w[0]/w[1]
    #
    # xx = np.linspace(-12, 34)
    # yy = a*xx-model.intercept_[0]/w[1]
    #
    # plt.plot(xx,yy, label ="logistic decision boundary")
    #
    # clf_lda = LinearDiscriminantAnalysis()
    # clf_lda.fit(X, Y)
    # w = clf_lda.coef_[0]
    # a = -w[0]/w[1]
    #
    # xx = np.linspace(-12, 34)
    # yy = a*xx-clf_lda.intercept_[0]/w[1]
    # plt.plot(xx,yy, color="blue", label ="LDA decision boundary")

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, color='b')
    plt.scatter(X[:, 0], X[:, 1], c=Y)

    plt.axis('tight')
    plt.legend()

    plt.show()

def plot_lda_only(filename, title, filename_fig):

    df = pd.read_csv(path+filename, names=['x1','x2','y'], header=None)
    fig = plt.figure()
    fig.suptitle(title, fontsize=20)
    columns_ls = []
    for column in df.columns:
        columns_ls.append(column)

    X = df[columns_ls[0:len(columns_ls)-1]].values
    Y = df[columns_ls[len(columns_ls)-1]].values

    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X, Y)
    w = clf_lda.coef_[0]
    a = -w[0]/w[1]

    xx = np.linspace(-12, 34)
    yy = a*xx-clf_lda.intercept_[0]/w[1]
    plt.plot(xx,yy, color="blue", label ="LDA decision boundary")

    print "Weights W0 %.2f and W1%.2f"%(w[0], w[1])
    plt.text(0, 0, "Y=+1", fontsize=12)
    plt.text(10, -20, "Y=-1", fontsize=12)
    # plt.plot(xx, yy_down, 'k--')
    # plt.plot(xx, yy_up, 'k--')
    # plt.plot(xx,yy,color="black", label ="svm decision boundary")


    plt.xlabel('X1', fontsize=18)
    plt.ylabel('X2', fontsize=16)
    # fig.savefig(filename_fig)
    # model = LogisticRegression()
    # model.fit(X, Y)
    # w = model.coef_[0]
    # a = -w[0]/w[1]
    #
    # xx = np.linspace(-12, 34)
    # yy = a*xx-model.intercept_[0]/w[1]
    #
    # plt.plot(xx,yy, label ="logistic decision boundary")
    #
    # clf_lda = LinearDiscriminantAnalysis()
    # clf_lda.fit(X, Y)
    # w = clf_lda.coef_[0]
    # a = -w[0]/w[1]
    #
    # xx = np.linspace(-12, 34)
    # yy = a*xx-clf_lda.intercept_[0]/w[1]
    # plt.plot(xx,yy, color="blue", label ="LDA decision boundary")

    # plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
    #         s=80, color='b')
    plt.scatter(X[:, 0], X[:, 1], c=Y)

    plt.axis('tight')
    plt.legend()

    plt.show()

def plot_svm_logistic(filename, title):

    df = pd.read_csv(path+filename, names=['x1','x2','y'], header=None)
    columns_ls = []
    for column in df.columns:
        columns_ls.append(column)

    X = df[columns_ls[0:len(columns_ls)-1]].values



    Y = df[columns_ls[len(columns_ls)-1]].values
    fig = plt.figure()
    fig.suptitle(title, fontsize=20)

    clf = svm.SVC(kernel='linear')

    clf.fit(X, Y)
    w = clf.coef_[0]
    print "Weights of SVM W0 = %.2f and W1 = %.2f"%(w[0], w[1])
    a = -w[0]/w[1]
    xx =np.linspace(-12, 34)
    yy = a*xx-clf.intercept_[0]/w[1]
    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    plt.text(0, 10, "Y=+1", fontsize=12)
    plt.text(10, 0, "Y=-1", fontsize=12)
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.plot(xx,yy,color="black", label ="svm decision boundary")

    model = LogisticRegression()
    model.fit(X, Y)
    w = model.coef_[0]
    a = -w[0]/w[1]
    print "Weights of Logistic regression W0 = %.2f and W1 = %.2f"%(w[0], w[1])
    xx = np.linspace(-12, 34)
    yy = a*xx-model.intercept_[0]/w[1]

    plt.plot(xx,yy, label ="logistic decision boundary")
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, color='b')
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.xlabel('X1', fontsize=18)
    plt.ylabel('X2', fontsize=16)
    plt.axis('tight')
    plt.legend()

    plt.show()
    # print clf.predict([.5, .7])

def plot_linear_svm_lda(filename, title):
    df = pd.read_csv(path+filename, names=['x1','x2','y'], header=None)
    columns_ls = []
    for column in df.columns:
        columns_ls.append(column)

    X = df[columns_ls[0:len(columns_ls)-1]].values
    Y = df[columns_ls[len(columns_ls)-1]].values
    fig = plt.figure()
    fig.suptitle(title, fontsize=20)

    clf = svm.SVC(kernel='linear')

    clf.fit(X, Y)
    w = clf.coef_[0]
    print "Weights SVM W0=%.2f and W1=%.2f"%(w[0], w[1])
    a = -w[0]/w[1]
    xx =np.linspace(-12, 34)
    yy = a*xx-clf.intercept_[0]/w[1]
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    # plot the line, the points, and the nearest vectors to the plane
    # plt.plot(xx, yy, 'k-')
    plt.text(0, 10, "Y=+1", fontsize=12)
    plt.text(10, 0, "Y=-1", fontsize=12)
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.plot(xx,yy,color="black", label ="svm decision boundary")

    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X, Y)
    w = clf_lda.coef_[0]
    a = -w[0]/w[1]
    print "Weights LDA W0=%.2f and W1=%.2f"%(w[0], w[1])
    xx = np.linspace(-12, 34)
    yy = a*xx-clf_lda.intercept_[0]/w[1]
    plt.plot(xx,yy, color="blue", label ="LDA decision boundary")

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, color='b')
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.xlabel('X1', fontsize=18)
    plt.ylabel('X2', fontsize=16)
    plt.axis('tight')
    plt.legend()
    plt.show()

plot_linear_svm_only("P3.txt", "Linear SVM", "Linear_SVM.jpg")
plot_linear_svm_only("P3_outliar.txt", "Linear SVM with outlier", "Linear_SVM_outlier.jpg")

plot_lda_only("P3.txt", "LDA classifier", "LDA_SVM.jpg")
plot_lda_only("P3_outliar.txt", "LDA Classifier With Outliers", "LDA_SVM.jpg")

plot_linear_svm_lda("P3.txt", "SVM with LDA comparision")
plot_linear_svm_lda("P3_outliar.txt", "SVM vs LDA with Outliers")

plot_svm_logistic("P3.txt", "SVM vs Logistic Regression")
plot_svm_logistic("P3_outliar.txt", "SVM vs Logistic Regression with Outliers")

plot_linear_svm("P3.txt", "SVM vs Logistic vs LDA")
plot_linear_svm("P3_outliar.txt", "SVM vs Logistic vs LDA with Outlier")

high_dimensional()






