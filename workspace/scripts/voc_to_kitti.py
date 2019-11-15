#voc_to_kitty.py
import collections
import os
from PIL import Image
from tqdm import tqdm_notebook as tqdm
import xmltodict
import argparse

# construct the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="input voc dataset path")
ap.add_argument("-wd", "--width", required=False, default='480',
    help="output image width")
ap.add_argument("-ht", "--height", required=False, default='320',
    help="output image height")
args = vars(ap.parse_args())

input_dir = args["input"]
image_width = int(args["width"]) #496
image_height = int(args["height"]) #320

def save_kitti_labels(label_list, output_file):
    """Function wrapper to save kitti format bbox labels to txt file.

    Args:
        label_list (list: list of string of kitti labels.
        output_file (str): path to output kitti label file.
    """
    with open(output_file, 'w') as f:
            if len(label_list) != 0:
                for line in label_list:
                    f.write(line)
    f.closed


def read_pascal_labels(xmlfile):
    """Function to read xml file and parse to return list of objects.

    Args:
        filename (str): path to the xml label file.
    """
    with open(xmlfile, 'r') as xfile:
        data = xmltodict.parse(xfile)

    object_list = data['annotation']['object']

    # Making sure that the outputs is always a list irrespective
    # of having just one object
    if isinstance(object_list, collections.OrderedDict):
        object_list = [object_list]

    return object_list


def map_to_kitti(object_list, sf):
    """Function to map xml objects to kitti object.

    Args:
        object_list: List of xml translated labels per object.
    """
    if not isinstance(object_list, list):
        raise TypeError("The object list is incompatible: {}".type(object_list))
    labels_list = []
    scale_x = sf[0]
    scale_y = sf[1]
    for element in object_list:
        class_name = element['name']
        xmin = float(element['bndbox']['xmin']) * scale_x
        ymin = float(element['bndbox']['ymin']) * scale_y
        xmax = float(element['bndbox']['xmax']) * scale_x
        ymax = float(element['bndbox']['ymax']) * scale_y
        label_tail = " 0.00 0.00 0.00 0.00 0.00 0.00 0.00\n"
        label_head = class_name.lower() + " 0.00 0 0.00 "
        bbox_string = "{:.3f} {:.3f} {:.3f} {:.3f}".format(xmin, ymin,
                                                           xmax, ymax)
        label_string = label_head + bbox_string + label_tail
        labels_list.append(label_string)
    return labels_list


def convert_to_kitti(voc_root, output_image_height, output_image_width):
    """Wrapper function to convert a VOC dataset to kitti format.

    Args:
        voc_root (str): Path to voc dataset root.
        output_image_height (int): image height for the output kitti images
        output_image_width (int): image_width for the output kitti images
    """
    output_size = (output_image_width, output_image_height)
    voc_labels_root = os.path.join(voc_root, "Annotations")
    kitti_trainval_labels_root = os.path.join(voc_root, "Annotations_kitti/trainval")
    kitti_test_labels_root = os.path.join(voc_root, "Annotations_kitti/test")
    voc_images_root = os.path.join(voc_root, "JPEGImages")
    kitti_trainval_images_root = os.path.join(voc_root, "JPEGImages_kitti/trainval")
    kitti_test_images_root = os.path.join(voc_root, "JPEGImages_kitti/test")
    if not os.path.exists(kitti_trainval_labels_root):
        os.makedirs(kitti_trainval_labels_root)

    if not os.path.exists(kitti_test_labels_root):
        os.makedirs(kitti_test_labels_root)

    if not os.path.exists(kitti_test_images_root):
        os.makedirs(kitti_test_images_root)

    if not os.path.exists(kitti_trainval_images_root):
        os.makedirs(kitti_trainval_images_root)

    labels = [os.path.splitext(item)[0] for item in sorted(os.listdir(voc_labels_root))
              if item.endswith('.xml')]

    for item in tqdm(labels):
        if "2011" in item:
            kitti_images_root = kitti_test_images_root
            kitti_labels_root = kitti_test_labels_root
        else:
            kitti_images_root = kitti_trainval_images_root
            kitti_labels_root = kitti_trainval_labels_root
        xmlfile = os.path.join(voc_labels_root, "{}.xml".format(item))
        kitti_file = os.path.join(kitti_labels_root, "{}.txt".format(item))
        image_file = os.path.join(voc_images_root, "{}.jpg".format(item))
        kitti_image = os.path.join(kitti_images_root, "{}.jpg".format(item))
        image = Image.open(image_file)
        resized_image = image.resize(output_size, Image.ANTIALIAS)
        resized_image.save(kitti_image)
        scale_x = float(output_image_width)/float(image.size[0])
        scale_y = float(output_image_height)/float(image.size[1])
        sf = (scale_x, scale_y)
        pascal_labels = read_pascal_labels(xmlfile)
        kitti_labels = map_to_kitti(pascal_labels, sf)
        save_kitti_labels(kitti_labels, kitti_file)


convert_to_kitti(input_dir, image_height, image_width)
