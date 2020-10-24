from scipy.ndimage.morphology import distance_transform_edt as dt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import scipy.signal

def createCircularImage(W,H,n,R):
    # parameter:
    # W, H  width and height of image
    # n     the number of blobs to generate
    # R     the radius of the blobs

    A = np.array([[1]*H]*W)

    if n > 0:
        x = random.choices(range(W), k=n)
        y = random.choices(range(H), k=n)
        A[x, y] = 0
        A = dt(A)
        A = A < R
    else:
        A = np.array([[0]*H]*W)

    return A.astype(int)

def createSquareImage(W, H, n):
    # parameter:
    # W, H  width and height of image
    # n     Number of squares to generate

    B = np.array([[0] * H] * W)

    if n > 0:
        x = random.choices(range(W), k=n)
        y = random.choices(range(H), k=n)

        # generate squares at random locations
        # Size are all 9*9, allow overlap
        l_l = 4
        for idx in range(n):
            tl = x[idx] - l_l
            if tl < 0: tl = 0
            tr = x[idx] + l_l
            if tr > W: tr = W
            bl = y[idx] - l_l
            if bl < 0: bl = 0
            br = y[idx] + l_l
            if br > H: br = H
            B[tl:tr + 1, bl:br + 1].fill(1)

    return B.astype(int)

def createCircularSquareImagePair(W, H, n):
    # parameter:
    # W, H  width and height of image
    # n     Number of circles/squares to generate

    # NOTE: This function only generated circular blobs with radius=5, and squares with size 9*9

    A = np.array([[1] * H] * W)
    B = np.array([[0] * H] * W)

    if n > 0:
        x = random.choices(range(W), k=n)
        y = random.choices(range(H), k=n)
        A[x, y] = 0
        A = dt(A)
        A = A < 5

        # generate squares at the same locations
        # R has to be 5
        l_l = 4
        for idx in range(n):
            tl = x[idx] - l_l
            if tl < 0: tl = 0
            tr = x[idx] + l_l
            if tr > W: tr = W
            bl = y[idx] - l_l
            if bl < 0: bl = 0
            br = y[idx] + l_l
            if br > H: br = H
            B[tl:tr + 1, bl:br + 1].fill(1)
    else:
        A = np.array([[0] * H] * W)

    return A.astype(int), B.astype(int)


def isOverlapped(img, N):
    # detect if a circle is 1) overlapped with other circles, or
    #                       2) is partially off the image
    return np.count_nonzero(img == 1) != int(69 * N)


def genBlobs(image_number, image_size, circle_radius, circle_max_number, allow_overlap, allow_square):
    # parameters:
    # image_number = 1000     # this is # of images
    # image_size = 64         # W = H
    # circle_radius = 5       # circle radius, DO NOT have effect when allow_square is enabled
    # circle_max_number = 9   # max number of circles that will appear

    training_set_circular = np.array([])
    if allow_square is True:
        training_set_square = np.array([])

    # generate # of training images
    for i in range(image_number):
        # number of blobs is random
        num_of_blobs = random.randint(1, circle_max_number)
        # create image

        if allow_square:
            img_circular, img_square = createCircularSquareImagePair(image_size, image_size, num_of_blobs)
        else:
            img_circular = createCircularImage(image_size, image_size, num_of_blobs, circle_radius)

        # only generate perfect cases
        if (allow_overlap is False) and (allow_square is False):
            while isOverlapped(img_circular, num_of_blobs):
                 img_circular = createCircularImage(image_size, image_size, num_of_blobs, circle_radius)
                 # print("overlapped, retry...")

        # add labels
        training_set_circular = np.append(training_set_circular, num_of_blobs)
        training_set_circular = np.append(training_set_circular, img_circular)

        if allow_square:
            training_set_square = np.append(training_set_square, num_of_blobs)
            training_set_square = np.append(training_set_square, img_square)

        # # plot image
        # print(img.shape)
        # print(img)
        # plt.imshow(img, cmap='Greys')
        # plt.show(block=True)
        print("{0}/{1}".format(i, image_number))

    if allow_square:
        # reshape square data
        training_set_square = training_set_square.reshape(image_number, (image_size * image_size + 1))
        # save data into .csv file
        np.savetxt("./data/my_data_square.csv", training_set_square.astype(int), fmt="%d", delimiter=",")

    # reshape circular data
    training_set_circular = training_set_circular.reshape(image_number, (image_size * image_size + 1))
    # save data into .csv file
    np.savetxt("./data/my_data.csv", training_set_circular.astype(int), fmt="%d", delimiter=",")


def genMixedBlobs(image_number, image_size, max_blob_number):
    # parameters:
    # image_number = 1000     # this is # of images
    # image_size = 64         # W = H

    training_set= np.array([])

    image_sub_num = int(image_number / 3)
    image_rem_num = int(image_number % 3)

    # generate circular (r=5) training images
    for i in range(image_sub_num):
        # number of blobs is random
        num_of_blobs = random.randint(1, max_blob_number)

        # create circular images with radius of 5
        img= createCircularImage(image_size, image_size, num_of_blobs, 5)

        # add labels
        training_set = np.append(training_set, num_of_blobs)
        training_set = np.append(training_set, img)

        # print current progress
        print("{0}/{1}, Circular r=5".format(i, image_number))

    # generate square training images
    for i in range(image_sub_num):
        # number of blobs is random
        num_of_blobs = random.randint(1, max_blob_number)

        # create square images with size 9*9
        img = createSquareImage(image_size, image_size, num_of_blobs)

        # add labels
        training_set = np.append(training_set, num_of_blobs)
        training_set = np.append(training_set, img)

        # print current progress
        print("{0}/{1}, Square".format(image_sub_num + i, image_number))

    # generate circular (r=6) training images
    for i in range(image_sub_num):
        # number of blobs is random
        num_of_blobs = random.randint(1, max_blob_number)

        # create circular images with radius of 6
        img = createCircularImage(image_size, image_size, num_of_blobs, 6)

        # add labels
        training_set = np.append(training_set, num_of_blobs)
        training_set = np.append(training_set, img)

        # print current progress
        print("{0}/{1}, Circular r=6".format(image_sub_num *2 + i, image_number))

    # generate the rest of the pictures, randomly choose from the previous types
    for i in range(image_rem_num):
        # number of blobs is random
        num_of_blobs = random.randint(1, max_blob_number)

        # type of image
        type = random.randint(1, 2)

        # create images depending on chosen type
        if type is 0:
            img = createCircularImage(image_size, image_size, num_of_blobs, 5)
        elif type is 1:
            img = createSquareImage(image_size, image_size, num_of_blobs)
        else:
            img = createCircularImage(image_size, image_size, num_of_blobs, 6)

        # add labels
        training_set = np.append(training_set, num_of_blobs)
        training_set = np.append(training_set, img)

        # print current progress
        print("{0}/{1}, Random type".format(image_sub_num * 3 + i, image_number))

    # reshape circular data
    training_set = training_set.reshape(image_number, (image_size * image_size + 1))
    # save data into .csv file
    np.savetxt("./data/my_data_mixed.csv", training_set.astype(int), fmt="%d", delimiter=",")


if __name__ == '__main__':
    #genBlobs(1000, 64, 6, 9, True, True)
    genMixedBlobs(10000, 64, 9)

    pass
