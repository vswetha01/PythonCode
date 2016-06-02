import argparse
import re
import os
import glob
import random
from math import log
from collections import Counter


# global counter object
positive_counter = Counter()
negative_counter = Counter()
total_pos_length = 0
total_neg_length = 0
unique_words_training_length = 0
# Stop word list
stopWords = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also',
             'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be',
             'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear',
             'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for',
             'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers',
             'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is',
             'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may',
             'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor',
             'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our',
             'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since',
             'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then',
             'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us',
             've', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
             'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet',
             'you', 'your', 's', 't', 're']


def parseArgument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('-d', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


# function to read files from a path
def filelist(pathspec):
    files_list = list()
    for name in glob.glob(pathspec):
        if (os.stat(name).st_size != 0):
            files_list.append(name)
    return files_list


# function to filter stop word and clean i/p
def words(doc):
    doc = re.sub('[^A-Za-z]+', ' ', doc)
    wordlist = doc.split(' ')
    wordlistCpy = []
    for word in wordlist:
        word = word.lower()
        if word not in stopWords and len(word) > 2:
            wordlistCpy.append(word)
    return wordlistCpy


# reads text from a file
def get_text(filename):
    f = open(filename, "r")
    f_txt = f.read()
    return f_txt


# create index for files and count all words in a particular class
def create_indexes(files, ispositive):
    tf_all_words = set()
    words_list_all = list()
    for f in files:
        file_txt = get_text(f)
        words_list = words(file_txt)
        words_list_all += words_list

    tf_all_words.update(words_list_all)
    if ispositive:
        global total_pos_length
        global positive_counter
        positive_counter.clear()
        positive_counter = Counter(words_list_all)
        total_pos_length = len(words_list_all)
    else:
        global total_neg_length
        global negative_counter
        negative_counter.clear()
        negative_counter = Counter(words_list_all)
        total_neg_length = len(words_list_all)

    return tf_all_words


# function to initialize +/- ve counter in training set
def do_training_set(files_pos, files_neg):
    tf_set = set()
    tf_pos_set = create_indexes(files_pos, True)
    tf_neg_set = create_indexes(files_neg, False)
    tf_set.update(tf_pos_set)
    tf_set.update(tf_neg_set)
    global unique_words_training_length
    # unique_words_training_length = len(tf_pos_set) + len(tf_neg_set)
    unique_words_training_length = len(tf_set)
    return


# function to return the freq of words in doc
def create_index_per_document(file):
    file_txt = get_text(file)
    words_list = words(file_txt)
    tf = Counter(words_list)
    return (tf)


# function to calculate the class probability
def get_class_probability(num_a, num_b):
    probability = 1.0 * num_a / float(num_a + num_b)
    return probability


# function to calculate positive probability of a document
def get_positive_probability(tf, ppos):
    sum = 0.0
    global positive_counter
    global total_pos_length
    global unique_words_training_length

    for t in tf:
        n = tf[t]
        wc = positive_counter[t]
        denom = total_pos_length + unique_words_training_length + 1
        numer = wc + 1
        pwc = float(numer) / float(denom)
        sum = sum + n * log(pwc) * 1.0
    prob_pos = ppos + sum
    return prob_pos


# calculate the negative probability of a document
def get_negative_probability(tf, pneg):
    sum = 0.0
    global negative_counter
    global total_neg_length
    for t in tf:
        n = tf[t]
        wc = negative_counter[t]
        denom = total_neg_length + unique_words_training_length + 1
        numer = wc + 1
        pwc = float(numer) / float(denom)
        sum = sum + n * log(pwc) * 1.0
    prob_neg = pneg + sum
    return prob_neg


# Process test set data
def do_test_set(files_pos, files_neg):
    count_p = 0
    count_n = 0
    prob_pos = get_class_probability(len(files_pos), len(files_neg))
    prob_neg = get_class_probability(len(files_neg), len(files_pos))
    # get probability for positive documents
    for file in files_pos:
        tf = create_index_per_document(file)
        prob_tf_pos = get_positive_probability(tf, prob_pos)
        prob_tf_neg = get_negative_probability(tf, prob_neg)
        pos = prob_pos + prob_tf_pos
        neg = prob_neg + prob_tf_neg
        if (pos > neg):
            count_p = count_p + 1

            # get probability for negative documents
    for file in files_neg:
        tf = create_index_per_document(file)
        prob_tf_pos = get_positive_probability(tf, prob_pos)
        prob_tf_neg = get_negative_probability(tf, prob_neg)
        pos = prob_pos + prob_tf_pos
        neg = prob_neg + prob_tf_neg
        if (neg > pos):
            count_n = count_n + 1

    return (count_p, count_n)


# function splits into test set and training set
def split_training_test_set(val1, val2, val3):
    test_set = []
    training_set = []

    training_set += val1
    training_set += val2

    test_set += val3
    return (test_set, training_set)


# function to calculate accuracy based on test and training
def calculate_accuracy(test_p, test_n, training_p, training_n):
    print "num_pos_test_docs: ", len(test_p)
    print "num_pos_training_docs: ", len(training_p)

    print "num_neg_test_docs: ", len(test_n)
    print "num_neg_training_docs: ", len(training_n)

    do_training_set(training_p, training_n)
    (count_p, count_n) = do_test_set(test_p, test_n)

    print "num_pos_correct_docs: ", count_p
    print "num_neg_correct_docs: ", count_n
    total_test_docs = len(test_p) + len(test_n)
    accuracy = 0.0
    accuracy = 100.0 * (count_p + count_n) / float(total_test_docs)
    print "accuracy: ", accuracy

    return accuracy


# function to split files into 3 parts
def split_files(path):
    file_list = []
    files = filelist(path)
    random.shuffle(files)
    lst_1 = files[0:len(files) / 3]
    lst_2 = files[len(files) / 3: 2 * len(files) / 3]
    lst_3 = files[2 * len(files) / 3:len(files)]

    file_list.append(lst_1)
    file_list.append(lst_2)
    file_list.append(lst_3)
    return file_list


def main():
    args = parseArgument()
    directory = args['d'][0]
    pos_path = "/pos/*.txt"
    p_list = []
    path = directory + pos_path
    p_list = split_files(path)

    n_list = []
    neg_path = "/neg/*.txt"
    path = directory + neg_path
    n_list = split_files(path)

    accuracy = 0.0
    for i in range(0, 3, 1):
        if i == 0:
            print "iteration: ", i + 1
            (test_lst_p, training_lst_p) = split_training_test_set(p_list[0], p_list[1], p_list[2])
            (test_lst_n, training_lst_n) = split_training_test_set(n_list[0], n_list[1], n_list[2])
            accuracy += calculate_accuracy(test_lst_p, test_lst_n, training_lst_p, training_lst_n)

        if i == 1:
            print "iteration: ", i + 1
            (test_lst_p, training_lst_p) = split_training_test_set(p_list[1], p_list[2], p_list[0])
            (test_lst_n, training_lst_n) = split_training_test_set(n_list[1], n_list[2], n_list[0])
            accuracy += calculate_accuracy(test_lst_p, test_lst_n, training_lst_p, training_lst_n)
        if i == 2:
            print "iteration: ", i + 1
            (test_lst_p, training_lst_p) = split_training_test_set(p_list[0], p_list[2], p_list[1])
            (test_lst_n, training_lst_n) = split_training_test_set(n_list[0], n_list[2], n_list[1])
            accuracy += calculate_accuracy(test_lst_p, test_lst_n, training_lst_p, training_lst_n)
    accuracy = float(accuracy) / 3.0
    print "Ave Accuracy: ", accuracy


main()
