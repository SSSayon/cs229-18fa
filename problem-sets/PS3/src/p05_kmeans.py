from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os


def init_centroids(num_clusters: int, image: np.ndarray) -> np.ndarray:
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of `image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    flat_image = image.reshape(-1, image.shape[-1])
    rand_idx = np.random.choice(flat_image.shape[0], size=num_clusters)
    return flat_image[rand_idx]
    # *** END YOUR CODE ***


def update_centroids(centroids: np.ndarray, image: np.ndarray, max_iter=30, print_every=10) -> np.ndarray:
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    # Usually expected to converge long before `max_iter` iterations
    # Initialize `dist` vector to keep track of distance to every centroid
    # Loop over all centroids and store distances in `dist`
    # Find closest centroid and update `new_centroids`
    # Update `new_centroids`
    H, W, _ = image.shape
    num_clusters = centroids.shape[0]
    centroids = centroids.astype(float)
    new_centroids = np.copy(centroids)
    it = 0
    idx = np.zeros((H, W), dtype=int)

    for it in range(max_iter):

        for i in range(H):
            for j in range(W):
                idx[i, j] = np.argmin(np.linalg.norm(image[i, j] - centroids, axis=1))

        for k in range(num_clusters):
            new_centroids[k] = np.mean(image[idx == k], axis=0) if np.any(idx == k) else centroids[k]
        
        if it % print_every == 0:
            loss = 0
            for i in range(H):
                for j in range(W):
                    loss += np.linalg.norm(new_centroids[idx[i, j]] - image[i, j]) ** 2
            loss /= (H * W)
            print(f"iteration: {it}, loss: {loss}")

        centroids = np.copy(new_centroids)

    # *** END YOUR CODE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    # Initialize `dist` vector to keep track of distance to every centroid
    # Loop over all centroids and store distances in `dist`
    # Find closest centroid and update pixel value in `image`
    H, W, _ = image.shape
    for i in range(H):
        for j in range(W):
            image[i, j] = centroids[np.argmin(np.linalg.norm(image[i, j] - centroids, axis=1))].astype(int)
    # *** END YOUR CODE ***

    return image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('../src/output', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('../src/output', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('../src/output', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    np.random.seed(229)
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='../data/peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='../data/peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
