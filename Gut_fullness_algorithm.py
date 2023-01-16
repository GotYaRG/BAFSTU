import numpy as np
import pandas as pd
from skimage import io, segmentation, feature, future, morphology, color, \
                    measure, filters, exposure, transform, draw
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import binary_fill_holes
from functools import partial
import os
import pickle


def validation_input_to_binary(img):
    """
    A simple function that converts annotated images made
    in photoshop to boolean type binary images.
    :param img: An annotated png image.
    :return: A binary image based on the input image.
    """
    blank_img = np.full((img.shape[0], img.shape[1]), False, dtype=bool)
    for y_i in range(img.shape[0]):
        for x_i in range(img.shape[1]):
            pxl = img[y_i, x_i]
            if pxl[3] != 0:
                blank_img[y_i, x_i] = True
            else:
                blank_img[y_i, x_i] = False
    return blank_img


def stretch_contrast(img):
    """
    A function that increases the contrast in an image by
    rescaling the intensity of the image based on the
    bottom and top 2% (in terms of intensity) of pixels.
    :param img: An rgb image.
    :return: An rgb image with increased contrast.
    """
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))


def dilater(object_to_dilate, rounds):
    """
    A function that takes an image and repeatedly dilates
    it, as many times as the user desires.
    :param object_to_dilate: The image that needs to be dilated.
    :param rounds: The amount of additional times the image
                   needs to be dilated.
    :return: The dilated image.
    """
    dilated_object = morphology.dilation(object_to_dilate)
    for i in range(rounds):
        dilated_object = morphology.dilation(dilated_object)
    return dilated_object


def check_borders(img):
    """
    A specific function used to check the amount of pixels on the
    border of a binary image that are True.
    :param img: A binary image.
    :return: The amount of border pixels set to True and the total amount
             of border pixels.
    """
    border_pixels, border_total = 0, 0
    for pxl in img[0][1:-1]:
        border_total += 1
        if pxl:
            border_pixels += 1
    for pxl_l in img:
        border_total += 1
        if pxl_l[0]:
            border_pixels += 1
        if pxl_l[-1]:
            border_pixels += 1
    for pxl in img[-1][1:-1]:
        border_total += 1
        if pxl:
            border_pixels += 1
    return border_pixels, border_total


def remove_border_pixels(edge):
    """
    A specific function that removes the pixels of a binary
    image that are on the very border of the image. Meant to
    be applied to edges found by an edge dectection tool.
    :param edge: The edge who's border pixels need removing.
    :return: The edge with its border pixels removed.
    """
    for y_i in range(edge.shape[0]):
        for x_i in range(edge.shape[1]):
            if y_i == 1 or x_i == 1 or y_i == edge.shape[0] - 2 or x_i == edge.shape[1] - 2:
                edge[y_i, x_i] = False
    return edge


def add_binary_pixels(img):
    """
    In order to obtain the contents of a Noctiluca we make use of "Trainable Segmentation" with a
    Random Forest classifier. This classifier can only be trained on a single image, but we wanted to train it
    on more than one image. In order to get around this, we can stitch multiple images onto eachother into one
    large image. However, for this to work, all images need to have either the same width or length. This function
    makes sure of exactly that, by extending the width of each image passed to it to 196 pixels, which is the width
    of the widest image in the training set.
    :param img: An image that might need to be widened.
    :return: An image that has a width of 196 pixels.
    """
    blank_image = np.full((img.shape[0], 196 - img.shape[1]),
                          False, dtype=bool)
    new_img = np.concatenate((img, blank_image), 1)
    return new_img


def add_black_pixels(img):
    """
    In order to obtain the contents of a Noctiluca we make use of "Trainable Segmentation" with a
    Random Forest classifier. This classifier can only be trained on a single image, but we wanted to train it
    on more than one image. In order to get around this, we can stitch multiple images onto eachother into one
    large image. However, for this to work, all images need to have either the same width or length. This function
    makes sure of exactly that, by extending the width of each image passed to it to 196 pixels, which is the width
    of the widest image in the training set.
    :param img: An image that might need to be widened.
    :return: An image that has a width of 196 pixels.
    """
    blank_image = np.full((img.shape[0], 196 - img.shape[1], 3),
                          0, dtype=np.uint8)
    new_img = np.concatenate((img, blank_image), 1)
    return new_img


def circle_finder(img):
    """
    A function that receives an image and is tasked with finding a circle of
    a particular size within that image. The size range of the circle is
    determined by the largest object that can be found in the image. The circle
    itself is calculated by running an edge detector on the image and feeding
    the resulting edges into a 'hough transform' function, together with the
    size range for the circle. The starting coordinates and radius of the best
    circle are returned.
    :param img: An image in which a circle has to be found.
    :return: The starting coordinates and radius of a circle.
    """
    # First, the object(s) in the image are labeled and the size of the largest
    # object is stored.
    labelled_blobs = measure.label(img)
    rp = measure.regionprops(labelled_blobs)
    size = int(round(max([i.axis_major_length for i in rp]) / 2, 0))

    # Next, the edges present edges are drawn and the best circle is extrapolated
    # from them.
    edges = feature.canny(img, sigma=3)
    hough_radii = np.arange(size, size*2, 2)
    hough_res = transform.hough_circle(edges, hough_radii)
    accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    return cx, cy, radii


def fill_partials(img):
    """
    A function that attempts to fill the surface of a Noctiluca
    individual that isn't entirely visible in the image. Usually, when
    there is a 'hole' in a collection of pixels, the Binary Fill function
    from scikit can fill the hole. However, when this hole is on the border
    of an image, it's not entirely enclosed by pixels. This causes the
    Binary Fill function to fill. Flood Fill from skImage will still work,
    however you need the right coordinates for where to start filling. This
    function does exactly that, by approximating the shape in a circle,
    then using a smaller circle as a reference for Flood Fill.

    :param img: A binary image with a convex shape that has holes in it.
    :return: The same binary image, with its holes filled in.
    """

    # First, the generally round shape of the individual is approximated
    # using hough transform in circe_finder() to find a circle of a certain size.
    cx, cy, radii = circle_finder(img)
    image = img.copy()
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = draw.circle_perimeter(center_y, center_x, radius,
                                             shape=image.shape)
        image[circy, circx] = True

    # After the approximate circular shape has been found, a much smaller
    # circle (1/3 of the original radius) is made.
    q_radius = round(radii[0] / 3, 0)
    y_up = cy[0] + q_radius
    y_down = cy[0] - q_radius
    x_up = cx[0] + q_radius
    x_down = cx[0] - q_radius

    # In order to not overfill the image, a check based on border pixels
    # is put in place for each flood fill coordinate. If too many border
    # pixels are flipped, overfilling probably occurred. When the check
    # is triggered, the flood fill won't take place at that coordinate.
    pre_flood_b_p, pre_flood_b_t = check_borders(img)
    flooded = img.copy()

    # The smaller circle that was calculated earlier is used here as a
    # reference for the coordinates needed to use Flood Fill.
    for y, x in zip([y_up, y_up, y_up, cy[0], cy[0], cy[0], y_down, y_down, y_down],
                    [x_up, cx[0], x_down, x_up, cx[0], x_down, x_up, cx[0], x_down]):
        try:
            post_flood_b_p, post_flood_b_t = check_borders(
                segmentation.flood_fill(flooded, (int(y), int(x)), True, connectivity=1))
            try:
                border_percent = (post_flood_b_t - post_flood_b_p) / (pre_flood_b_t - pre_flood_b_p)
            except ZeroDivisionError:
                border_percent = 0
            if border_percent > 0.15:
                flooded = segmentation.flood_fill(flooded, (int(y), int(x)), True, connectivity=1)
            else:
                pass
        except IndexError:
            pass
    return flooded


def remove_surface_junk(surface):
    """
    A function that attempts to cut off some of the artifacts that can
    show up on the images. From time to time an object will overlap with
    an individual and stick out, to more accurately calculate the surface
    of an individual, the part that sticks out would need to be removed.
    The problem is that it's hard to do so while not removing any of the
    actual surface. This function attempts to do so using hough transform.
    :param surface:
    :return:
    """

    # First, Convex Hull is used to see if there is something that sticks
    # out from the shape in the image. If nothing sticks out, and the
    # surface has its normal smooth-ish convex shape, the hull will
    # fit around it very neatly. If something sticks out, it won't.
    hull = morphology.convex_hull_image(surface)
    surface_c, hull_c = 0, 0
    for l1, l2 in zip(surface, hull):
        for p1, p2 in zip(l1, l2):
            if p1:
                surface_c += 1
            if p2:
                hull_c += 1
    increase = 100 - (surface_c / hull_c * 100)

    # If there is an increase in pixels of more than 7.5% in the convex hull
    # it is assumed that there is an artifact of some kind sticking out
    # from the surface.
    if increase > 7.5:
        cx, cy, radii = circle_finder(surface)
        # The convex shape of the individual is approximated using hough
        # transform in circle_finder. The radius of the resulting circle is
        # increased by 10% to be sure the entire individual is included in
        # the circle. Anything outside the circle is removed.
        circle = np.full((surface.shape[0], surface.shape[1]), False, dtype=bool)
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = draw.circle_perimeter(center_y, center_x, int(radius*1.1), shape=circle.shape)
            circle[circy, circx] = True
        filled_circle = segmentation.flood_fill(circle, (cy[0], cx[0]), True, connectivity=1)
        filtered = np.full((surface.shape[0], surface.shape[1]),
                           False, dtype=bool)
        for y in range(surface.shape[0]):
            for x in range(surface.shape[1]):
                if surface[y, x] and filled_circle[y, x]:
                    filtered[y, x] = True
        return filtered
    else:
        return surface


def surface_and_outline(img):
    """
    A function that takes an image of N. scintillans and attempts
    to determine the "surface" and outline of the organism.

    :param img: an RGB image of an N. scintillans individual.
    :return: The surface and dilated outline of the individual.
    """
    # First the contrast is increased, to make it easier to
    # seperate the background from the individual.
    contrast_img = stretch_contrast(img)

    # Next the image is "thresholded" to perform the actual
    # seperation of background and foreground.
    gray_img = color.rgb2gray(contrast_img)
    threshold = filters.threshold_triangle(gray_img)
    th_img = gray_img > threshold

    # Sometimes, an image can be somewhat overexposed. Still
    # increasing the contrast on such an image would result
    # in most of the foreground being caught in the treshold.
    # This is checked by the following If statement, which, if
    # triggered, thresholds the image once more without
    # increasing the contrast beforehand.
    if np.sum(th_img) / (th_img.shape[0]*th_img.shape[1]) > 0.9:
        gray_img = color.rgb2gray(img)
        threshold = filters.threshold_triangle(gray_img)
        th_img = gray_img > threshold

    # Often times, parts of the inside of the individual are dark enough
    # to be seen as background, causing them to be removed by thresholding.
    # The three steps below ensure that these 'holes' are filled in
    # most cases.
    closed_img = morphology.binary_closing(th_img)
    filled_th_img = binary_fill_holes(closed_img)
    filled_th_img = fill_partials(filled_th_img)

    # Often there are small objects or tiny speckles in frame that remain
    # after thresholding. To remove these, we label all individual objects
    # and remove everything that is smaller than the largest object.
    labelled_blobs = measure.label(filled_th_img)
    rp = measure.regionprops(labelled_blobs)
    size = max([i.area for i in rp])
    big = morphology.remove_small_objects(filled_th_img, min_size=size - 2000)

    # From time to time, a piece of detritus or some other object overlaps
    # with an individual, which is then often thresholded with the rest of
    # the foreground. This usually results in an object sticking out of the
    # surface, making it seem bigger than it actually is. To somewhat
    # alleviate this, the function below tests how round the individual is
    # with a Convex Hull, and if it's not round enough it attempts to cut
    # off some of the pixels that make it less round.
    surface = remove_surface_junk(big)

    # With the surface finally determined, we can also easily obtain
    # the outline from the surface. The outline is heavily dilated as well,
    # for later use as a mask for the cell wall.
    outline = feature.canny(surface, sigma=3)
    outline = remove_border_pixels(outline)
    dilated_outline = dilater(outline, 15)
    return surface, dilated_outline


def get_trained_segmentation_images():
    """
    A function that imports all the images required for the training of the Random Forest classifier,
    which is used to segement the Noctiluca images.
    :return: A dictionary of labeled images, each linked to a reference image by filename.
    """
    img_dict = {}
    path = "Data/photoshop/exports/trained_segmentation_exports/"
    for img_name in os.listdir(path):
        img = io.imread(path + img_name)
        img_binary = validation_input_to_binary(img)
        img_addition = add_binary_pixels(img_binary)
        img_code = img_name[0:19]
        if img_code not in img_dict:
            img_dict[img_code] = [img_addition]
        else:
            img_dict[img_code].append(img_addition)
    return img_dict


def stitch_labels_and_train_predictor(img_dict):
    """
    In order to obtain the contents of a Noctiluca we make use of "Trainable Segmentation" with a
    Random Forest classifier. This function recieves a dictionary of images that each contain a different type
    of pixel, namely Background, Food, Nucleus or Membrane pixels. These pixel type images are merged into one
    template that is linked to a reference RGB image that contains the original image. The pixel type templates
    and original images are fed to the Random Forest classifier, in order to train it on which pixel in the
    original image is of which type. The result is a trained classifier capable of predicting pixel types in new images.
    :param img_dict: A dictionary of images with differently labeled pixels.
    :return: A trained Random Forest classifier.
    """
    training_labels, features = [], []
    sigma_min, sigma_max = 1, 16
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=False, texture=False,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            channel_axis=-1)

    for img_name, labels_list in img_dict.items():
        try:
            img = io.imread("data/noctiluca_sample/" + img_name + ".0.png")
        except FileNotFoundError:
            img = 0
        if type(img) == int:
            try:
                img = io.imread("data/noctiluca_sample/" + img_name + ".1.png")
            except FileNotFoundError:
                img = io.imread("data/noctiluca_sample/" + img_name + ".2.png")

        img = add_black_pixels(img)

        training_label = np.zeros(img.shape[:2], dtype=np.uint8)

        # Background labels
        training_label[labels_list[0]] = 1

        # Detritus labels
        training_label[labels_list[1]] = 2

        if len(labels_list) == 4:
            # Nucleosome labels
            training_label[labels_list[3]] = 3

            # Membrane labels
            training_label[labels_list[2]] = 4
        else:
            # Nucleosome labels
            training_label[labels_list[2]] = 3

        training_labels.append(training_label)
        features.append(features_func(img))

    concat_label, concat_feature = 0, 0
    for label, feature_var in zip(training_labels, features):
        if type(concat_label) == int:
            concat_label = label
            concat_feature = feature_var
        else:
            concat_label = np.concatenate((concat_label, label), 0)
            concat_feature = np.concatenate((concat_feature, feature_var), 0)

    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                 max_depth=10, max_samples=0.05)
    clf = future.fit_segmenter(concat_label, concat_feature, clf)
    predictor = future.TrainableSegmenter(clf, features_func)
    return predictor


def isolate_contents(img, outline, surface):
    """
    The random forest classifier used to segment the images into their respective pixel types is not perfect.
    For one, we are really only interested in the pixels which are deemed "Food". But from time to time, the
    classifier thinks a piece of food is actually a nucleus, which can happen due to a number of circumstances.
    This can result in multiple nuclei being found, when there is (almost) never more than one. This function
    seperates the "Food" pixels from the rest and attempts to isolate a single nucleus, reverting other nucleus
    pixels back to food pixels.
    :param img: An image which has been segmented into four different types of pixels.
    :param outline: The outline of the object in the image, drastically dilated.
    :param surface: The entire surface of the object in the image.
    :return: The food pixels that were found in the object.
    """
    if len(set(list(img.flatten()))) == 3:
        nuc_layer = np.full((img.shape[0], img.shape[1]), False, dtype=bool)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if img[y, x] == 3:
                    nuc_layer[y, x] = True

        labelled_blobs = measure.label(nuc_layer)
        rp = measure.regionprops(labelled_blobs)
        try:
            size = max([i.area for i in rp])
            biggest = morphology.remove_small_objects(nuc_layer, min_size=size - 10)
        except ValueError:
            biggest = nuc_layer.copy()
        biggest = dilater(biggest, 2)

        new_img = np.full((img.shape[0], img.shape[1]), False, dtype=bool)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if not biggest[y, x] and not outline[y, x] and surface[y, x] and img[y, x] == 2:
                    new_img[y, x] = True
                if not biggest[y, x] and not outline[y, x] and surface[y, x] and img[y, x] == 3:
                    new_img[y, x] = True
    else:
        new_img = np.full((img.shape[0], img.shape[1]), False, dtype=bool)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if not outline[y, x] and surface[y, x] and img[y, x] == 2:
                    new_img[y, x] = True
    return new_img


def load_usable_images():
    """
    There are lots of N. scintillans images available, however, not all
    of them are usable by this algorithm. For this reason, the images
    are classified into different categories by a Neural Network.
    The output of this NN is stored in a dataframe, which this function
    imports. After importing the dataframe, only the usable images are
    returned.
    :return: A dataframe containing image_names of which this algorithm
             can calculate the gut fullness.
    """
    # A good input location still needs to be set
    in_file = open("Data/NN_output/prediction.pickle", "rb")
    new_df = pickle.load(in_file)
    in_file.close()
    is_digesting = new_df["label"] == "digesting"
    digesting_df = new_df[is_digesting]
    return digesting_df["image_name"]


def export_gut_fullness_percentages(gut_fullness_percentages, usable_images):
    """
    In order to use the calculated gut fullness percentages elsewhere, they are
    stored in a dataframe and exported as a pickle. This allows them to be easily
    imported and/or stored elsewhere for other applications.
    :param gut_fullness_percentages: A list of gut fullness percentages, in
                                     parallel to the names of the usable images.
    :param usable_images: A dataframe containing the names of usable images.
    """
    out_df = pd.DataFrame()
    out_df["image_name"] = usable_images
    out_df["gut fullness"] = gut_fullness_percentages
    # A good output location still needs to be set
    pd.to_pickle(out_df, "Data/gut_fullness_output/gut_fullness_percentages.pickle")


def main():
    # Load images the algorithm can use
    usable_images = load_usable_images()

    # Load images for the Random Forest Classifier and train the predictor
    trained_segmentation_images = get_trained_segmentation_images()
    predictor = stitch_labels_and_train_predictor(trained_segmentation_images)

    gut_fullness_percentages, usable_images_list = [], []

    for img_name in usable_images:
        usable_images_list.append(img_name)
        # Load images one by one and calculate surface and contents
        img = io.imread("Data/images/" + img_name)
        surface, outline = surface_and_outline(img)
        segmented = predictor.predict(img)
        contents = isolate_contents(segmented, outline, surface)

        # Calculate gut fullness using the surface and cell contents
        gut_fullness = np.sum(contents) / np.sum(surface) * 100
        gut_fullness_percentages.append(gut_fullness)

    # Export the gut fullness percentages of the images as a dataframe
    export_gut_fullness_percentages(gut_fullness_percentages, usable_images_list)


if __name__ == '__main__':
    main()
