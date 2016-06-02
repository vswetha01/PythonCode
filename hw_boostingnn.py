import numpy as np
from sklearn.datasets import make_hastie_10_2
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

def boosting():
    X, y = make_hastie_10_2(random_state=0)
    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]

    # implement boosting in python

    num_of_loops = np.arange(0, 500, 50)
    # print num_of_loops
    test_errors = []
    N = len(X_train)

    w = np.ones(N)/N
    models = []
    alphas = []
    train_errors = []
    for num_of_iterations in num_of_loops:
        print num_of_iterations
        for i in range(num_of_iterations):
            clf = DecisionTreeClassifier(max_depth=1)
            model = clf.fit(X_train,y_train, sample_weight=w)
            pred = clf.predict(X_train)
            # print "sum of misclassification is %d"%sum(pred != y_train)
            numerator = np.dot(w, (pred!= y_train))

            # denominator = sum(w)
            error_m = numerator #/float(denominator)
            val = (1. - error_m)/float(error_m)
            alpha_m = (np.log(1 - error_m) - np.log(error_m))/2.

            w = w * np.exp(alpha_m * (pred != y_train))
            w = w/float(w.sum())
            models.append(model)
            alphas.append(alpha_m)
        # Compute train error
        tr_length = len(y_train)
        tr_result = np.zeros(tr_length)
        for m, al in zip(models, alphas):
            tr_result = tr_result + al*m.predict(X_train)
        tr_result = np.sign(tr_result)
        tr_error = np.sum(tr_result != y_train)/float(tr_length)
        train_errors.append(tr_error)


        # Compute test error
        M = len(y_test)
        result = np.zeros(M)

        for m,al in zip(models, alphas):
            result = result + al * m.predict(X_test)

        result = np.sign(result)
        test_error = np.sum(result != y_test)/float(M)
        test_errors.append(test_error)

    plt.xlabel("Boosting iterations")
    plt.ylabel("Test/Train error")
    plt.plot(num_of_loops, test_errors, "k--")
    plt.plot(num_of_loops, train_errors, label ="train error")
    # plt.legend([p1,p2],["p1","p2"])
    plt.text(100, .1, "Training error", fontsize=12, color="R")
    plt.text(100, .4, "Test error", fontsize=12, color="Black")

    plt.show()


def gbm():
    X, y = make_hastie_10_2(random_state=0)
    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]
    n_all = []
    test_errors = []
    train_errors = []
    for i in range(100):
        n = 10*(i+1)
        n_all.append(n)
        clf = GradientBoostingClassifier(n_estimators=n, learning_rate=.5, \
                                     max_depth=2, random_state=0).fit(X_train, y_train)

        test_error = clf.score(X_test, y_test)
        test_errors.append(test_error)
        train_error = clf.score(X_train, y_train)
        train_errors.append(train_error)
        print n, train_error, test_error
    plt.xlabel("Boosting iterations")
    plt.ylabel("Test (Red)/ train(Blue) accuracy")
    plt.text(10, .95, "Training accuracy", fontsize=12, color="B")
    plt.text(50, 0.87, "Test accuracy", fontsize=12, color="R")
    plt.plot(n_all, test_errors)
    plt.plot(n_all, train_errors)
    plt.legend()
    plt.show()

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

# X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
# y = np.array([[0],[1],[1],[0]])
#
# # syn0 = 2*np.random.random((3,4)) - 1
#
# np.random.seed(1)
# syn0 = 2*np.random.random((3,4))-1
# syn1 = 2*np.random.random((4,1))-1
#
# for j in range(100):
#     # feed forward through layer 0, layer1, layer2
#     l0 = X
#     l1 = (np.dot(l0,syn0))
#     l2 = (np.dot(l1, syn1))
# #     print l2, np.shape(l2)
#     # actual error
#     #l2_error = -2*(y - l2)
#     l2_error = y - l2
#
#     if (j%2) ==0:
#         print "Error:"+str(np.mean(np.abs(l2_error)))
#     l2_delta = l2_error*nonlin(l2,deriv=True) #-2*(y - l2)
#     l1_error = l2_delta.dot(syn1.T)
#     l1_delta = l1_error* nonlin(l1,deriv=True)
#     syn1 += l1.T.dot(l2_delta)
#     syn0 += l0.T.dot(l1_delta)

boosting()