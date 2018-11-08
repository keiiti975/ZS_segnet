import numpy as np
from PIL import Image
import os
import os.path
import matplotlib.pyplot as plt


def make_vectors(filename):
    """Create semantic_vector array"""
    vector_array = []

    text_file = open(filename, 'r')
    lines = text_file.readlines()
    text_file.close()

    for line in lines:
        line = line.rstrip()
        # input all attribute
        vector1 = line.split(" ")
        # remove index
        vector2 = vector1[1:]
        vector_array.append(vector2)

    vector_array = np.array(vector_array, 'float32')
    return vector_array


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def main():
    vector_array = make_vectors('./v_class/class1.txt')
    length = np.sum(vector_array, axis=1) / vector_array.shape[1]
    # print(length)
    avg_array = np.ones(vector_array.shape[1])
    similarity = np.zeros(vector_array.shape[0])
    for i in range(vector_array.shape[0]):
        similarity[i] = cos_sim(vector_array[i], avg_array)
    # print(similarity)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(similarity, length)
    ax1.set_title('semantic vector distribution')
    ax1.set_xlabel('cos_similarity')
    ax1.set_ylabel('vector length')
    fig1.savefig('figure1.png')

    std = np.std(vector_array, axis=1)
    vector_array2 = vector_array.copy()
    for i in range(vector_array.shape[0]):
        vector_array2[i] = (vector_array[i] - length[i]) / std[i]
        vector_array2[i] = vector_array2[i] + length[i]
    length = np.sum(vector_array2, axis=1) / vector_array.shape[1]
    # print(length)
    for i in range(vector_array.shape[0]):
        similarity[i] = cos_sim(vector_array2[i], avg_array)
    # print(similarity)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(similarity, length)
    ax2.set_title('semantic vector distribution')
    ax2.set_xlabel('cos_similarity')
    ax2.set_ylabel('vector length')
    fig2.savefig('figure2.png')

    vector_array3 = vector_array.copy()
    for i in range(vector_array.shape[0]):
        vector_array3[i] = sigmoid(vector_array[i])
    length = np.sum(vector_array3, axis=1) / vector_array.shape[1]
    # print(length)
    for i in range(vector_array.shape[0]):
        similarity[i] = cos_sim(vector_array3[i], avg_array)
    # print(similarity)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.scatter(similarity, length)
    ax3.set_title('semantic vector distribution')
    ax3.set_xlabel('cos_similarity')
    ax3.set_ylabel('vector length')
    fig3.savefig('figure3.png')


if __name__ == '__main__':
    main()
