

from CustomJson import CustomJson
from GTJSONReaderMuret import GTJSONReaderMuret
import sys, os, re
import shutil
import cv2
import numpy as np
from os.path import dirname
import copy


def createParentDirectory(path_file):
    pathdir = dirname(path_file)
    mkdirp(pathdir)

def mkdirp(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

def deleteFolder(directory):
    if os.path.isdir(directory):
        shutil.rmtree(directory, ignore_errors=True)



def listFilesRecursive(path_dir):
    
    try:
        listOfFile = os.listdir(path_dir)
    except Exception:
        pathdir_exec = os.path.dirname(os.path.abspath(__file__))
        path_dir = pathdir_exec + "/" + path_dir
        listOfFile = os.listdir(path_dir)

    list_files = list()
    
    for entry in listOfFile:
        fullPath = os.path.join(path_dir, entry)
        if os.path.isdir(fullPath):
            list_files = list_files + listFilesRecursive(fullPath)
        else:
            list_files.append(fullPath)
    
    list_files.sort()            
    return list_files


def match_SRC_GT_Images(list_src_images, list_gt_images):
    list_matched_data = []
    for idx_image in range(len(list_src_images)):
        src_image = list_src_images[idx_image]
        gt_image = list_gt_images[idx_image]

        src_basename = os.path.basename(src_image)
        gt_basename = os.path.basename(gt_image).replace(".json", "")

        print('*'*80)
        print("Image %d:" % (idx_image))
        print("\t%s" % (src_image))
        print("\t%s" % (gt_image))

        assert(src_basename == gt_basename)

        list_matched_data.append( (src_image, gt_image))
        

    print("SRC and GT images are match.")
    return list_matched_data


def get_List_boundingBoxes_from_JSON_with_sequences(json_pathfile, num_annotated_regions, considered_classes=["staff", "empty-staff"]):
    js = CustomJson()
    js.loadJson(json_pathfile)

    gtjson = GTJSONReaderMuret()
    gtjson.load(js)

    if num_annotated_regions == -1:
        bboxes_with_sequences = gtjson.getListBoundingBoxesAndSymbols(considered_classes=considered_classes)
    else:
        bboxes_with_sequences = gtjson.getListBoundingBoxesAndSymbols(considered_classes=considered_classes)[0:num_annotated_regions]

    return bboxes_with_sequences


def get_List_boundingBoxes_from_JSON(json_pathfile, num_annotated_regions, considered_classes=["staff", "empty-staff"]):
    js = CustomJson()
    js.loadJson(json_pathfile)

    gtjson = GTJSONReaderMuret()
    gtjson.load(js)

    if num_annotated_regions == -1:
        bboxes = gtjson.getListBoundingBoxes(considered_classes=considered_classes)
    else:
        bboxes = gtjson.getListBoundingBoxes(considered_classes=considered_classes)[0:num_annotated_regions]

    return bboxes

def generar_mascara(bounding_boxes, matriz_etiquetas):
    # Inicializar la máscara con ceros
    mascara = np.zeros_like(matriz_etiquetas)

    # Iterar sobre los bounding boxes de tipo 1
    for bbox in bounding_boxes:
        x_min, y_min, x_max, y_max = bbox

        # Obtener la región de interés (ROI) de la matriz etiquetas
        roi = matriz_etiquetas[y_min:y_max, x_min:x_max]

        # Verificar si hay algún píxel de tipo 2 en la ROI
        if np.any(roi == 2):
            # Si hay píxeles de tipo 2, reducir el bounding box
            mascara[y_min:y_max, x_min:x_max] = 2
        else:
            # Si no hay píxeles de tipo 2, establecer el bounding box completo
            mascara[y_min:y_max, x_min:x_max] = 1

    return mascara

def expand_masking_bboxes2(bboxes, gt_img):
    gt_masking = np.zeros_like(gt_img)
    gt_hidden = (gt_img == 2)

    for bbox in bboxes:
        x_start, y_start, x_end, y_end = bbox

        # Expandir en la dimensión x
        while x_start >= 0 and not np.any(gt_hidden[x_start:x_end, y_start:y_end]):
            x_start -= 1
        x_start += 1

        while x_end < gt_hidden.shape[0] and not np.any(gt_hidden[x_start:x_end, y_start:y_end]):
            x_end += 1
        x_end -= 1

        # Expandir en la dimensión y
        while y_start >= 0 and not np.any(gt_hidden[x_start:x_end, y_start:y_end]):
            y_start -= 1
        y_start += 1

        while y_end < gt_hidden.shape[1] and not np.any(gt_hidden[x_start:x_end, y_start:y_end]):
            y_end += 1
        y_end -= 1

        gt_masking[x_start:x_end, y_start:y_end] = 1

    return gt_masking

def expand_masking_bboxes(bboxes, gt_img):
    gt_masking = np.zeros((gt_img.shape[0], gt_img.shape[1]))
    gt_hidden = (gt_img==2)
    for bbox in bboxes:
        x_start_orig = bbox[0]
        x_end_orig = bbox[2]
        y_start_orig = bbox[1]
        y_end_orig = bbox[3]
        
        x_start = bbox[0]
        x_end = bbox[2]
        y_start = bbox[1]
        y_end = bbox[3]

        #TOP
        while(x_start>=0 and np.sum(gt_hidden[x_start:x_start_orig, y_start:y_end]) == 0 ):
            x_start-=1
        if (x_start -1) < x_start_orig:
            x_start += 1

        #BOTTOM
        while(x_end<gt_hidden.shape[0] and np.sum(gt_hidden[x_end_orig:x_end, y_start:y_end]) == 0 ):
            x_end+=1
        if x_end > (x_end_orig+1):
            x_end-=1
        
        while(y_start>=0 and np.sum(gt_hidden[x_start:x_end, y_start:y_start_orig]) == 0 ):
            y_start-=1
        if (y_start-1) < y_start_orig:
            y_start+=1
        
        while(y_end<gt_hidden.shape[1] and np.sum(gt_hidden[x_start:x_end, y_end:y_end_orig]) == 0 ):
            y_end+=1
        if y_end > y_end_orig+1:
            y_end-=1
        
        gt_masking[x_start:x_end, y_start:y_end] = 1
    return gt_masking
        
def get_Masking(json_pathfile, nb_annotated_regions, img_shape, considered_classes=["staff", "empty-staff"]):
    js = CustomJson()
    js.loadJson(json_pathfile)
    gtjson = GTJSONReaderMuret()
    gtjson.load(js)

    gt_im, num_annotated_regions = gtjson.generateGT(considered_classes=considered_classes, img_shape = img_shape, vertical_reduction_regions=0.2, nb_annotated_regions=nb_annotated_regions)
    
    bboxes = get_List_boundingBoxes_from_JSON(json_pathfile, nb_annotated_regions, considered_classes)
    if nb_annotated_regions == -1 or nb_annotated_regions == num_annotated_regions:
        masking = np.ones((gt_im.shape[0], gt_im.shape[1]))
    else:
        masking = expand_masking_bboxes(bboxes, gt_im)
        
    return masking

def load_gt_image(json_pathfile, nb_annotated_regions, img_shape, vertical_reduction_regions, considered_classes=["staff", "empty-staff"]):

    js = CustomJson()
    js.loadJson(json_pathfile)

    gtjson = GTJSONReaderMuret()
    gtjson.load(js)

    gt_im, num_annotated_regions = gtjson.generateGT(considered_classes=considered_classes, img_shape = img_shape, vertical_reduction_regions=vertical_reduction_regions, nb_annotated_regions=nb_annotated_regions)

    return gt_im==1, num_annotated_regions


def load_src_image(path_file, mode=cv2.IMREAD_COLOR):

    file_img = cv2.imread(path_file, mode)
    if file_img is None : 
        raise Exception(
            'It is not possible to load the image\n'
            "Path: " + str(path_file)
        )

    return file_img


def readListStringFiles(list_pathfile):
    content_files = ""
    for path_file in list_pathfile:
        content_files += readStringFile(path_file)

    return content_files

def readStringFile(path_file):
    assert type(path_file) == str

    f = open(path_file)
    
    content = f.read()
    f.close()
    
    assert type(content) == str

    return content


def getSRCfile_from_JSONfile(json_pathfile):
    return json_pathfile.replace(".json", "").replace("JSON/", "SRC/")

def getList_SRCfiles_from_JSONfiles(list_json_files):
    if list_json_files is None:
        return []
    list_src_files = []
    for json_file in list_json_files:
        src_file = getSRCfile_from_JSONfile(json_file)
        list_src_files.append(src_file)

    return list_src_files


def readListStringFile(path_file):
    lines = readStringFile(path_file)
    list_lines = lines.split("\n")
    
    list_filtered = [line for line in list_lines if line != ""]

    return list_filtered

def saveImage (image, path_file):
    assert 'numpy.ndarray' in str(type(image))
    assert type(path_file) == str
    
    path_dir = dirname(path_file)

    if not os.path.exists(path_dir):
        os.makedirs(path_dir, 493)

    cv2.imwrite(path_file, image)


def appendString(content_string, path_file, close_file = True):
    assert type(content_string) == str
    assert type(path_file) == str
    
    path_dir = dirname(path_file)

    if not os.path.exists(path_dir):
        os.makedirs(path_dir, 493)
            
    f = open(path_file,"a")
    f.write(content_string + "\n")
    
    if close_file == True:
        f.close()
    
def writeString(content_string, path_file, close_file = True):
    assert type(content_string) == str
    assert type(path_file) == str
    
    path_dir = dirname(path_file)

    if not os.path.exists(path_dir):
        os.makedirs(path_dir, 493)
            
    f = open(path_file,"w")
    f.write(content_string + "\n")
    
    if close_file == True:
        f.close()
    
        
        
def __remove_attribute_namespace(config, key):
    try:
        delattr(config, key)
    except:
        pass

def getPathModel(config):
    config_copy = copy.deepcopy(config)
    
    __remove_attribute_namespace(config_copy, 'test')
    
    __remove_attribute_namespace(config_copy, 'db_test_txt')
    __remove_attribute_namespace(config_copy, 'gpu')
    __remove_attribute_namespace(config_copy, 'verbose')
    __remove_attribute_namespace(config_copy, 'res')
    __remove_attribute_namespace(config_copy, 'save')
    __remove_attribute_namespace(config_copy, 'aug_test')
    __remove_attribute_namespace(config_copy, 'n_aug')
    __remove_attribute_namespace(config_copy, 'drop_test')
    __remove_attribute_namespace(config_copy, 'iou')
    __remove_attribute_namespace(config_copy, 'path_model')
    

    if config_copy.standard == False:
        __remove_attribute_namespace(config_copy, 'standard')
    
    if config_copy.adapt_size == False:
        __remove_attribute_namespace(config_copy, 'adapt_size')
    
    if config.no_mask is None or config.no_mask == False:
        __remove_attribute_namespace(config_copy, 'no_mask')
        

    
    str_config = str(config_copy).replace("Namespace", "modelCNN_").replace("(", "").replace(")", "").replace("=", "_").replace("'", "").replace(",","").replace(" ", "__").replace("[", "_").replace("]","_").replace("]","_").replace("/", "_")
    str_config = "models/modelCNN/"+str_config + ".h5"
    str_config = str_config.replace("datasets","dbs").replace("training", "train").replace("db_train","dtr").replace("pages_train", "pt")
    str_config = str_config.replace("Folds", "").replace("fold", "fd")
    str_config = str_config.replace("staff", "st").replace("empty", "emp").replace("text", "tx").replace("lyrics","lyr")
    str_config = str_config.replace("Mus-Tradicional", "MT").replace("m-combined", "m").replace("c-combined", "c")
    str_config = str_config.replace("standard", "std")
    str_config = str_config.replace("combined", "cmb").replace("adapt_size", "adp").replace("no_mask", "nmsk")
    str_config = str_config.replace("flip","f")
    str_config = str_config.replace("std_True","std")
    str_config = str_config.replace("_nmsk_True_", "_nmsk_")
    
    return str_config