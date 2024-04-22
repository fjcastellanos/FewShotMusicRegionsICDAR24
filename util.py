import utilIO
import random
import numpy as np
import utilConst
import tensorflow as tf
import cv2
from operator import itemgetter
import FscoreRegion
from CustomJson import CustomJson
from GTJSONReaderMuret import GTJSONReaderMuret
import copy


def order_corpora_by_resolution(corpora, increase=True):
    list_pages_with_total_pixels = []
    
    for page in corpora:
        page_src = page[0]
        im = utilIO.load_src_image(page_src)
        resolution = im.shape
        total_pixels = resolution[0]*resolution[1]
        list_pages_with_total_pixels.append( (page, total_pixels) )
        
    list_pages_with_total_pixels = sorted(list_pages_with_total_pixels, key=itemgetter(1), reverse = increase) #Descendent order of total pixels

    list_pages_ordered_without_total_pixels = [page for (page, total_pixels) in list_pages_with_total_pixels]

    return list_pages_ordered_without_total_pixels


def create_Validation_and_Training_partitions(list_src_train, list_gt_train, list_src_val, list_gt_val, pages_train=None):
    
    train_corpora = utilIO.match_SRC_GT_Images(list_src_train, list_gt_train)
    val_corpora = utilIO.match_SRC_GT_Images(list_src_val, list_gt_val)

    train_corpora = order_corpora_by_resolution(train_corpora)
    val_corpora = order_corpora_by_resolution(val_corpora)
    

    if pages_train is None or pages_train == -1:
        pages_train = len(train_corpora)

    num_val_images = min(len(val_corpora),pages_train)
    
    train_data = train_corpora[0:pages_train]
    val_data = val_corpora[0: num_val_images]
    
    return train_data, val_data


def getInputShape(config):
    return (config.win_w, config.win_w, utilConst.kNUMBER_CHANNELS)


def appendNewSample(gr, gt, row, col, patch_height, patch_width, gr_chunks, gt_chunks):
    gr_sample = gr[
            row : row + patch_height, col : col + patch_width
        ]  # Greyscale image
    gt_sample = gt[
        row : row + patch_height, col : col + patch_width
    ]  # Ground truth
    gr_chunks.append(gr_sample)
    gt_chunks.append(gt_sample)



def extractSequentialSamplesClass(gr, gt, window_w, window_h, batch_size, idx_starting_patch, gr_chunks, gt_chunks, regions_mask, augmentation_types):
    ROWS = gt.shape[0]
    COLS = gt.shape[1]
    
    min_rate_annotated_pixels = 0.

    patch_counter = 0

    #print("extractSequentialSamplesClass")
    #print("Starting: " + str(idx_starting_patch))
    
    for row in range(window_w//2, ROWS+window_w//2-1, window_w):
        for col in range(window_h//2, COLS+window_h//2-1, window_h):
            row = min(row, ROWS-window_w//2)
            col = min(col, COLS-window_h//2)
            
            gr_sample = gr[
                row-window_w//2 : row-window_w//2 + window_w, col-window_h//2 : col-window_h//2 + window_h
            ]
            gt_sample = gt[
                row-window_w//2 : row-window_w//2 + window_w, col-window_h//2 : col-window_h//2 + window_h
            ]
            regions_mask_sample = regions_mask[
                row-window_w//2 : row-window_w//2 + window_w, col-window_h//2 : col-window_h//2 + window_h
            ]
            
            if (np.sum(gt_sample == 1) > batch_size):
                current_rate_annotated_pixels = np.sum(gt_sample == 1) / (window_h*window_w)
                            
                if current_rate_annotated_pixels > min_rate_annotated_pixels:
                    if patch_counter >= idx_starting_patch:
                        #print (patch_counter)
                        #print (str(row) + "-" + str(col))
                    
                        gr_aug_sample, gt_aug_sample, regions_mask_aug_sample, applied_augmentations = apply_random_augmentations(gr_sample, gt_sample, regions_mask_sample, augmentation_types, window_w, window_h)
                        gr_chunks.append(gr_aug_sample)
                        gt_chunks.append(gt_aug_sample)
                    
                    patch_counter += 1

                    if patch_counter >=(idx_starting_patch + batch_size):
                        return patch_counter
    return patch_counter
        

def extractRandomSamplesClass(gr, gt, patch_width, patch_height, batch_size, gr_chunks, gt_chunks, regions_mask, augmentation_types):

    potential_training_examples = np.where(gt == 1)

    num_coords = len(potential_training_examples[0])
    
    tries = 0
    MAX_TRIES = 100

    if num_coords >= batch_size:
        num_samples = 0
        while (num_samples < batch_size):
            idx_coord = random.randint(0, num_coords-1)
            row = potential_training_examples[0][idx_coord]
            col = potential_training_examples[1][idx_coord]
            
            offset_window_row = random.randint(-patch_width//2.2, patch_width//2.2)
            offset_window_col = random.randint(-patch_height//2.2, patch_height//2.2)
            
            row += offset_window_row
            col += offset_window_col

            row = max(patch_width//2+1, row)
            col = max(patch_height//2+1, col)

            row = min(gr.shape[0]-patch_width//2-1, row)
            col = min(gr.shape[1]-patch_height//2-1, col)

            gr_sample = gr[
                row-patch_width//2 : row-patch_width//2 + patch_width, col-patch_height//2 : col-patch_height//2 + patch_height
            ]
            gt_sample = gt[
                row-patch_width//2 : row-patch_width//2 + patch_width, col-patch_height//2 : col-patch_height//2 + patch_height
            ]
            regions_mask_sample = regions_mask[
                row-patch_width//2 : row-patch_width//2 + patch_width, col-patch_height//2 : col-patch_height//2 + patch_height
            ]

            gr_aug_sample, gt_aug_sample, regions_mask_aug_sample, applied_augmentations = apply_random_augmentations(gr_sample, gt_sample, regions_mask_sample, augmentation_types, patch_width, patch_height)
            
            current_rate_annotated_pixels = np.sum(gt_aug_sample == 1) / (patch_height*patch_width)
            if current_rate_annotated_pixels > 0 or tries > MAX_TRIES:
                gr_chunks.append(gr_aug_sample)
                gt_chunks.append(gt_aug_sample)
                num_samples+=1
                tries = 0
            else:
                tries+=1
    else:
        print("No annotated pixels found...")
        raise Exception("No annotated pixels found...")
        

def apply_mask(gt_img, regions_mask=None):
    if regions_mask is not None:
        masked = np.logical_and(gt_img, regions_mask)*1
        return masked
    else:
        return gt_img

def calculate_annotated_samples(gr, gt, window_w, window_h, nb_sequential_patches = -1):
    ROWS = gt.shape[0]
    COLS = gt.shape[1]
    
    patch_counter = 0
    
    X_samples = []
    Y_samples = []
    for row in range(window_w//2, ROWS+window_w//2-1, window_w):
        for col in range(window_h//2, COLS+window_h//2-1, window_h):
            row = min(row, ROWS-window_w//2)
            col = min(col, COLS-window_h//2)
            
            gt_sample = gt[row-window_w//2:row-window_w//2+window_w, col-window_h//2:col-window_h//2+window_h]
            gr_sample = gr[row-window_w//2:row-window_w//2+window_w, col-window_h//2:col-window_h//2+window_h]
            
            
            if (np.sum(gt_sample == 1) > 0):
                current_rate_annotated_pixels = np.sum(gt_sample == 1) / (window_h*window_w)
                            
                if nb_sequential_patches == -1 or (nb_sequential_patches == 0 and current_rate_annotated_pixels > 0) or current_rate_annotated_pixels > 0:
                    X_samples.append(gr_sample)
                    Y_samples.append(gt_sample)

                    patch_counter += 1

                    if nb_sequential_patches != -1 and nb_sequential_patches != 0 and patch_counter >=nb_sequential_patches:
                        return X_samples, Y_samples

    return X_samples, Y_samples


def calculate_mask(gt, window_w, window_h, nb_sequential_patches = -1):
    ROWS = gt.shape[0]
    COLS = gt.shape[1]
    mask = np.zeros((ROWS, COLS))

    patch_counter = 0
    
    for row in range(window_w//2, ROWS+window_w//2-1, window_w):
        for col in range(window_h//2, COLS+window_h//2-1, window_h):
            row = min(row, ROWS-window_w//2)
            col = min(col, COLS-window_h//2)
            
            gt_sample = gt[row-window_w//2:row-window_w//2+window_w, col-window_h//2:col-window_h//2+window_h]
            
            
            if (np.sum(gt_sample == 1) > 0):
                current_rate_annotated_pixels = np.sum(gt_sample == 1) / (window_h*window_w)
                            
                if nb_sequential_patches == -1 or (nb_sequential_patches == 0 and current_rate_annotated_pixels > 0) or current_rate_annotated_pixels > 0:
                    
                    mask[row-window_w//2:row-window_w//2+window_w, col-window_h//2:col-window_h//2+window_h] = 1
                    patch_counter += 1

                    if nb_sequential_patches != -1 and nb_sequential_patches != 0 and patch_counter >=nb_sequential_patches:
                        return mask, patch_counter

    return mask, patch_counter
    

def getListPages_with_content(list_data, considered_classes, number_annotated_regions):
    list_data_with_information = []

    nb_annotated_regions_already_annotated = 0
    for item_data in list_data:
        page_gt = item_data[1]
        js = CustomJson()
        js.loadJson(page_gt)

        gtjson = GTJSONReaderMuret()
        gtjson.load(js)
        if number_annotated_regions == -1:
            number_regions_to_be_annotated = -1
        else:    
            number_regions_to_be_annotated = number_annotated_regions - nb_annotated_regions_already_annotated
        num_annotated_regions = gtjson.get_number_annotated_regions(considered_classes=considered_classes, nb_annotated_regions=number_regions_to_be_annotated)

        nb_annotated_regions_already_annotated += num_annotated_regions
        if num_annotated_regions > 0:
            list_data_with_information.append(item_data)
        if nb_annotated_regions_already_annotated == number_annotated_regions:
            break

    return list_data_with_information

def getListDataWithContent(list_data, considered_classes):
    
    list_data_with_information = []
    for item_data in list_data:
        page_src = item_data[0]
        page_gt = item_data[1]
        gr = utilIO.load_src_image(page_src)
        img_shape = gr.shape[0:2]
        js = CustomJson()
        js.loadJson(page_gt)

        gtjson = GTJSONReaderMuret()
        gtjson.load(js)

        _, num_annotated_regions = gtjson.generateGT(considered_classes=considered_classes, img_shape = img_shape, vertical_reduction_regions=0, nb_annotated_regions=-1)
        if num_annotated_regions > 0:
            list_data_with_information.append(item_data)

    return list_data_with_information


def get_number_annotated_regions_page(json_pathfile, nb_annotated_regions, considered_classes):
    
    js = CustomJson()
    js.loadJson(json_pathfile)

    gtjson = GTJSONReaderMuret()
    gtjson.load(js)

    num_annotated_regions = gtjson.get_number_annotated_regions(considered_classes=considered_classes, nb_annotated_regions=nb_annotated_regions)

    return num_annotated_regions


def get_gt_image_and_regions(gt_path_file, nb_annotated_regions, img_shape, vertical_reduction_regions, with_expanding_bboxes, considered_classes=["staff", "empty-staff"]):

    gt_img, num_annotated_regions = (utilIO.load_gt_image(gt_path_file, nb_annotated_regions, img_shape, vertical_reduction_regions, considered_classes))
    
    if with_expanding_bboxes:
        regions_mask = utilIO.get_Masking(gt_path_file, nb_annotated_regions, img_shape, considered_classes)
    else:
        regions_mask = np.copy(gt_img)
    
    #regions_mask, n_patches = calculate_mask(gt_img, window_w, window_h, nb_annotated_regions)
    #gt_img = apply_mask(gt_img, regions_mask=regions_mask)

    return gt_img, regions_mask, num_annotated_regions


def normalize_image(img):
    return (255.-img) / 255.


def extract_list_annotated_samples(page_src, page_gt, nb_annotated_patches, window_w, window_h, vertical_reduction_regions, with_expanding_bboxes, considered_classes=["staff", "empty-staff"]):
    gr = utilIO.load_src_image(page_src)
    gr = normalize_image(gr)
    img_shape = (gr.shape[0], gr.shape[1])
    gt, _, _ = get_gt_image_and_regions(page_gt, nb_annotated_patches, img_shape, vertical_reduction_regions, with_expanding_bboxes, considered_classes)
    

    X_samples, Y_samples = calculate_annotated_samples(gr, gt, window_w, window_h, nb_annotated_patches)

    return X_samples, Y_samples


def get_image_with_gt(page_src, page_gt, nb_annotated_regions, vertical_reduction_regions, with_expanding_bboxes, considered_classes=["staff", "empty-staff"], with_mask=False):
    gr = utilIO.load_src_image(page_src)
    gr = normalize_image(gr)
    img_shape = (gr.shape[0], gr.shape[1])
    gt, regions_mask, nb_annotated_regions_real = get_gt_image_and_regions(page_gt, nb_annotated_regions, img_shape, vertical_reduction_regions, with_expanding_bboxes, considered_classes)
    

    if with_mask:
        #Deactivate the training process for pixels outside the region mask
        l = np.where((regions_mask == 0))
        gr[l] = utilConst.kPIXEL_VALUE_FOR_MASKING
    else:
        regions_mask = np.ones((gt.shape[0], gt.shape[1]))

    return gr, gt, regions_mask, nb_annotated_regions_real


def dump_image_with_size(gr, gt, regions_mask, width_out, height_out):

    ROWS=gr.shape[0]
    COLS=gr.shape[1]

    center_w = ROWS // 2
    center_h = COLS // 2

    if (len(gr.shape) == 3):
        gr_new = np.ones((width_out, height_out, gr.shape[2]))*(-1)
    else:
        gr_new = np.ones((width_out, height_out))*(-1)
    gt_new = np.zeros((width_out, height_out))
    regions_mask_new = np.zeros((width_out, height_out))

    rows_to_copy = min(ROWS, width_out)
    cols_to_copy = min(COLS, height_out)

    center_w = ROWS // 2
    center_h = COLS // 2
    center_w_new = width_out // 2
    center_h_new = height_out // 2

    gr_new[center_w_new-rows_to_copy//2:center_w_new-rows_to_copy//2 + rows_to_copy, center_h_new-cols_to_copy//2:center_h_new-cols_to_copy//2 + cols_to_copy] = gr[center_w-rows_to_copy//2:center_w-rows_to_copy//2+rows_to_copy, center_h-cols_to_copy//2:center_h-cols_to_copy//2+cols_to_copy]
    if gt is not None:
        gt_new[center_w_new-rows_to_copy//2:center_w_new-rows_to_copy//2 + rows_to_copy, center_h_new-cols_to_copy//2:center_h_new-cols_to_copy//2 + cols_to_copy] = gt[center_w-rows_to_copy//2:center_w-rows_to_copy//2+rows_to_copy, center_h-cols_to_copy//2:center_h-cols_to_copy//2+cols_to_copy]
    regions_mask_new[center_w_new-rows_to_copy//2:center_w_new-rows_to_copy//2 + rows_to_copy, center_h_new-cols_to_copy//2:center_h_new-cols_to_copy//2 + cols_to_copy] = regions_mask[center_w-rows_to_copy//2:center_w-rows_to_copy//2+rows_to_copy, center_h-cols_to_copy//2:center_h-cols_to_copy//2+cols_to_copy]

    return gr_new, gt_new, regions_mask_new


def apply_random_augmentations(gr, gt, regions_mask, augmentation_types, width_out, height_out):

    gr_aug = gr
    gt_aug = gt
    regions_mask_aug = regions_mask
    applied_augmentations = []

    augmentation_types_aux = augmentation_types
    if "random" in augmentation_types:
        augmentation_types_aux = [item for item in augmentation_types if item != "random"]
        if len(augmentation_types_aux) == 0:
            augmentation_types_aux.append("none")
        
    random.shuffle(augmentation_types_aux)
    
    for augmentation_type in augmentation_types_aux:
        activate_augmentation = random.randint(0, 1) == 1

        if activate_augmentation:
            gr_aug, gt_aug, regions_mask_aug, type_augmentation_out = apply_augmentation(gr_aug, gt_aug, regions_mask_aug, augmentation_type)
            applied_augmentations.append(type_augmentation_out)

    gr_new, gt_new, regions_mask_new =  dump_image_with_size(gr_aug, gt_aug, regions_mask_aug, width_out, height_out)
    return gr_new, gt_new, regions_mask_new, applied_augmentations


def calculateScaleFactor_to_adapt_image_to_patches(data_pages, window_h, nb_annotated_regions, considered_classes):
    
    list_region_heights = []
    for page in data_pages:
        gt_bboxes = utilIO.get_List_boundingBoxes_from_JSON(page[1], nb_annotated_regions, considered_classes)

        for gt_bbox in gt_bboxes:
            region_height = gt_bbox[2] - gt_bbox[0]
            list_region_heights.append(region_height)
    
    average_region_height = np.average(list_region_heights)
    
    scale_factor = float(window_h//2) / average_region_height
    
    return scale_factor



def getImages_with_scaleFactor(data_pages, adapt_size, scale_factor_adapt_size, nb_annotated_regions, vertical_reduction_regions, considered_classes):
    list_grs = []
    list_gts = []
    list_region_mask = []
    
    for page in data_pages:
        gr, gt, regions_mask, n_annotated_patches_real = get_image_with_gt(page[0], page[1], nb_annotated_regions, vertical_reduction_regions, True, considered_classes, True)
        if adapt_size:
            gr = image_resize_given_ratio(gr, ratio_width = None, ratio_height = scale_factor_adapt_size)
            gt = image_resize_given_ratio(gt, ratio_width = None, ratio_height = scale_factor_adapt_size)
            regions_mask = image_resize_given_ratio(regions_mask, ratio_width = None, ratio_height = scale_factor_adapt_size)
            gt = gt>0.5
        
        list_grs.append(gr)
        list_gts.append(gt)
        list_region_mask.append(regions_mask)
        
    return list_grs, list_gts, list_region_mask
    
    
def getRandomSamples_Given_Image(gr, gt, regions_mask, batch_size, window_w, window_h, augmentation_types):
    gr_chunks = []
    gt_chunks = []
 
    while len(gr_chunks) < batch_size:
        extractRandomSamplesClass(gr, gt, window_w, window_h, 1, gr_chunks, gt_chunks, regions_mask, augmentation_types)

    gr_chunks_arr = np.array(gr_chunks)
    gt_chunks_arr = np.array(gt_chunks)
    gt_chunks_arr = np.reshape(gt_chunks_arr, (gt_chunks_arr.shape[0], gt_chunks_arr.shape[1], gt_chunks_arr.shape[2], 1))
    # convert gr_chunks and gt_chunks to the numpy arrays that are yield below    

    return gr_chunks_arr, gt_chunks_arr
    

                            
def getRandomSamples(page, adapt_size, scale_factor_adapt_size, batch_size, nb_annotated_regions, window_w, window_h, augmentation_types, vertical_reduction_regions, considered_classes=["staff", "empty-staff"]):
    gr_chunks = []
    gt_chunks = []
 
    gr, gt, regions_mask, n_annotated_patches_real = get_image_with_gt(page[0], page[1], nb_annotated_regions, vertical_reduction_regions, True, considered_classes, True)
    
    if adapt_size:
        gr = image_resize_given_ratio(gr, ratio_width = None, ratio_height = scale_factor_adapt_size)
        gt = image_resize_given_ratio(gt, ratio_width = None, ratio_height = scale_factor_adapt_size)
        regions_mask = image_resize_given_ratio(regions_mask, ratio_width = None, ratio_height = scale_factor_adapt_size)
        gt = gt>0.5
        
    
    if n_annotated_patches_real == 0:
        return None, None
    
    while len(gr_chunks) < batch_size:
        extractRandomSamplesClass(gr, gt, window_w, window_h, 1, gr_chunks, gt_chunks, regions_mask, augmentation_types)

    gr_chunks_arr = np.array(gr_chunks)
    gt_chunks_arr = np.array(gt_chunks)
    gt_chunks_arr = np.reshape(gt_chunks_arr, (gt_chunks_arr.shape[0], gt_chunks_arr.shape[1], gt_chunks_arr.shape[2], 1))
    # convert gr_chunks and gt_chunks to the numpy arrays that are yield below    

    return gr_chunks_arr, gt_chunks_arr


#https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize_given_ratio(image, ratio_width = None, ratio_height = None):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if ratio_width is None and ratio_height is None:
        return image

    # check to see if the width is None
    if ratio_width is None:
        r = ratio_height
        dim = (int(w * r), int(h * r))

    # otherwise, the height is None
    else:
        r = ratio_width
        dim = (int(w * r), int(h * r))

    # resize the image
    resized = cv2.resize(image*1.0, dim)

    # return the resized image
    return resized

def resizeImage(img, new_size):

    old_size = img.shape[:2] # old_size is in (height, width) format


    ratio_w = float(new_size[0])/old_size[0]
    ratio_h = float(new_size[1])/old_size[1]
    
    ratio = min(ratio_w, ratio_h)
    new_size_with_aspect_ratio = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(img*1.0, (new_size_with_aspect_ratio[1], new_size_with_aspect_ratio[0]))

    delta_w = abs(new_size_with_aspect_ratio[1] - new_size[1])
    delta_h = abs(new_size_with_aspect_ratio[0] - new_size[0])
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im

def resizeBack(img, old_size):
    current_size = img.shape

    ratio_w = float(current_size[0])/old_size[0]
    ratio_h = float(current_size[1])/old_size[1]
    
    ratio = min(ratio_w, ratio_h)
    current_size_with_aspect_ratio = tuple([int(x*ratio) for x in old_size])

    # current_size should be in (width, height) format
    

    delta_w = abs(current_size_with_aspect_ratio[1] - current_size[1])
    delta_h = abs(current_size_with_aspect_ratio[0] - current_size[0])
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    im_crop = img[top:img.shape[0]-bottom, left:img.shape[1]-right]

    new_im = cv2.resize(im_crop*1.0, (old_size[1], old_size[0]))
    
    return new_im

def getResizedImages(pages, nb_annotated_regions, window_w, window_h, vertical_reduction_regions, considered_classes=["staff", "empty-staff"]):
    gr_chunks = []
    gt_chunks = []
    region_mask_chunks = []
    
    nb_annotated_regions_local = 0

    for page in pages:
        if nb_annotated_regions == -1:
            nb_annotated_regions_to_be_annotated = -1
        else:    
            nb_annotated_regions_to_be_annotated = nb_annotated_regions-nb_annotated_regions_local
        
        if nb_annotated_regions_to_be_annotated == -1 or nb_annotated_regions_to_be_annotated > 0:
            gr, gt, regions_mask, n_annotated_patches_real = get_image_with_gt(page[0], page[1], nb_annotated_regions, vertical_reduction_regions, True, considered_classes, True)
            nb_annotated_regions_local += n_annotated_patches_real

            gr_resized = cv2.resize(gr*1.0, (window_h, window_w)) #resizeImage(gr, (window_w, window_h))
            gt_resized = cv2.resize(gt*1.0, (window_h, window_w)) #resizeImage(gt*1, (window_w, window_h))
            gt_resized = gt_resized==1
            regions_mask_resized = cv2.resize(regions_mask*1.0, (window_h, window_w))
			#resizeImage(regions_mask*1, (window_w, window_h))
            regions_mask_resized = regions_mask_resized==1

            if np.amax(gt_resized) == 1:
                gr_chunks.append(gr_resized)
                gt_chunks.append(gt_resized)
                region_mask_chunks.append(regions_mask_resized)
    
    # convert gr_chunks and gt_chunks to the numpy arrays that are yield below    

    return gr_chunks, gt_chunks, region_mask_chunks



def getSequentialSamples(gr, gt, regions_mask, idx_patch, batch_size, n_annotated_patches_real, nb_annotated_patches, window_w, window_h, augmentation_types):
    
    #print("Annotated:")
    #print(n_annotated_patches_real)
    patch_counter = 0
    gr_chunks = []
    gt_chunks = []    
    patch_counter = idx_patch
    while len(gr_chunks) < batch_size and patch_counter < min(n_annotated_patches_real, nb_annotated_patches):
        patch_counter = extractSequentialSamplesClass(gr, gt, window_w, window_h, 1, patch_counter, gr_chunks, gt_chunks, regions_mask, augmentation_types)
        if len(gr_chunks) == 0:
            print ("Is none")
            return None
        
    gr_chunks_arr = np.array(gr_chunks)
    gt_chunks_arr = np.array(gt_chunks)
    gt_chunks_arr = np.reshape(gt_chunks_arr, (gt_chunks_arr.shape[0], gt_chunks_arr.shape[1], gt_chunks_arr.shape[2], 1))
    # convert gr_chunks and gt_chunks to the numpy arrays that are yield below    
    return gr_chunks_arr, gt_chunks_arr



def get_number_annotated_patches(page, nb_annotated_regions=-1, considered_classes=["staff", "empty-staff"]):
    
    if type(page) is tuple:
        n_annotated_patches_real_total = get_number_annotated_regions_page(page[1], nb_annotated_regions, considered_classes)
    else:
        n_annotated_patches_real_total = 0
        n_annotated_patches_real = 0
        nb_annotated_regions_to_be_annotated = nb_annotated_regions
        for p in page:
            if nb_annotated_regions == -1:
                nb_annotated_regions_to_be_annotated = -1
            else:
                nb_annotated_regions_to_be_annotated -= n_annotated_patches_real
            n_annotated_patches_real = get_number_annotated_regions_page(p[1], nb_annotated_regions_to_be_annotated, considered_classes)
            n_annotated_patches_real_total += n_annotated_patches_real
            if n_annotated_patches_real_total == nb_annotated_regions:
                break
    return n_annotated_patches_real_total

def get_number_pages_with_content(pages, nb_annotated_regions=-1, considered_classes=["staff", "empty-staff"]):
    number_pages = 0
    n_annotated_patches_real_total = 0
    n_annotated_patches_real = 0
    nb_annotated_regions_to_be_annotated = nb_annotated_regions
    for p in pages:
        if nb_annotated_regions == -1:
            nb_annotated_regions_to_be_annotated = -1
        else:
            nb_annotated_regions_to_be_annotated -= n_annotated_patches_real
        n_annotated_patches_real = get_number_annotated_regions_page(p[1], nb_annotated_regions_to_be_annotated, considered_classes)
        if n_annotated_patches_real > 0:
            number_pages += 1
        n_annotated_patches_real_total += n_annotated_patches_real
        if n_annotated_patches_real_total == nb_annotated_regions:
            break
    return number_pages

def create_generator(data_pages, no_mask, batch_size, window_shape, nb_patches, nb_annotated_patches, augmentation_types, vertical_reduction_regions, considered_classes=["staff", "empty-staff"]):
    if no_mask is None or no_mask == False:
        using_mask = True
    else:
        using_mask = False 

    idx_tries = 0
    while(True):
        #print("Shuffle training data...")
        random.shuffle(data_pages)
        #print("Done")
        
        for page in data_pages:
            if utilConst.AUGMENTATION_RANDOM in augmentation_types:
                assert(nb_patches != -1)

                gr_chunks_arr, gt_chunks_arr = getRandomSamples(page, False, None, min(batch_size, nb_patches), nb_annotated_patches, window_shape[0], window_shape[1], augmentation_types, vertical_reduction_regions, considered_classes)
                if gr_chunks_arr is None and gt_chunks_arr is None:
                    idx_tries+=1
                    if idx_tries > len(data_pages):
                        error_msg = 'It is not possible to annotate samples'
                        print(error_msg)
                        raise Exception(error_msg)
                    continue
                else:
                    idx_tries=0
                    yield gr_chunks_arr, gt_chunks_arr
            else:
                assert(nb_annotated_patches == nb_patches)
                real_patches = get_number_annotated_patches(page, window_shape[0], window_shape[1], nb_annotated_patches, considered_classes)
                if nb_annotated_patches == -1:
                    nb_annotated_patches_real = real_patches
                    np_patches_real = real_patches
                else:
                    nb_annotated_patches_real = nb_annotated_patches
                    np_patches_real = nb_patches
                gr, gt, regions_mask, n_annotated_patches_real = get_image_with_gt(page[0], page[1], nb_annotated_patches_real, vertical_reduction_regions, True, considered_classes, using_mask)
                idx_patch = 0
                while idx_patch < min(n_annotated_patches_real, nb_annotated_patches_real):    
                    samples = getSequentialSamples(gr, gt, regions_mask, idx_patch, min(batch_size, real_patches), n_annotated_patches_real, n_annotated_patches_real, window_shape[0], window_shape[1], augmentation_types)
                    if samples is not None:
                        idx_patch += len(samples[0])
                        yield samples[0], samples[1]
                    else:
                        idx_patch = min(n_annotated_patches_real, nb_annotated_patches)
                        



def create_generator_renewed(data_pages, standard, adapt_size, no_mask, batch_size, window_shape, nb_patches, nb_annotated_patches, augmentation_types, vertical_reduction_regions, considered_classes=["staff", "empty-staff"]):
    if no_mask is None or no_mask == False:
        using_mask = True
    else:
        using_mask = False 

    idx_tries = 0
    scaleFactor = calculateScaleFactor_to_adapt_image_to_patches(data_pages, window_shape[1], nb_annotated_patches, considered_classes)
    if adapt_size:
        print("Scale factor to adapt to the windows: " + str(scaleFactor))
    else:
        print("Scale factor is not applied.")

    while(True):
        #print("Shuffle training data...")
        #random.shuffle(data_pages)
        #print("Done")
        
        if standard:
            list_grs, list_gts, list_region_mask = getResizedImages(data_pages, nb_annotated_patches, window_shape[0], window_shape[1], vertical_reduction_regions, considered_classes)
            while(True):
                list_grs_copy = copy.deepcopy(list_grs)
                list_gts_copy = copy.deepcopy(list_gts)
                list_region_mask_copy = copy.deepcopy(list_region_mask) 
                
                randomlist = random.sample(range(0, len(list_grs_copy)), len(list_grs_copy))

                while len(randomlist) > 0:
                    
                    randomlist_loc = randomlist[0:batch_size]
                    randomlist = randomlist[batch_size:]
                
                    gr_list = []
                    gt_list = []
                    for idx_elem in randomlist_loc:
                        gr_elem = list_grs_copy[idx_elem] 
                        gt_elem = list_gts_copy[idx_elem] 
                        region_mask_elem = list_region_mask_copy[idx_elem] 
                        gr_aug, gt_aug, regions_mask_aug, type_augmentation_out = apply_random_augmentations(gr_elem, gt_elem, region_mask_elem, augmentation_types, window_shape[0], window_shape[1])
                        gr_list.append(gr_aug)
                        gt_list.append(gt_aug)

                    gr_batch = np.array(gr_list)
                    gt_batch = np.array(gt_list)
                    gt_batch = np.reshape(gt_batch, (gt_batch.shape[0], gt_batch.shape[1], gt_batch.shape[2], 1))
                    # convert gr_chunks and gt_chunks to the numpy arrays that are yield below    

                    yield gr_batch, gt_batch

        else:
            
                list_grs, list_gts, list_region_mask = getImages_with_scaleFactor(data_pages, adapt_size, scaleFactor, nb_annotated_patches, vertical_reduction_regions, considered_classes)
                
                while(True):
                    
                    randomlist = random.sample(range(0, len(list_grs)), len(list_grs))
                    
                    gr_list = []
                    gt_list = []
                    for idx_elem in randomlist:
                        gr_elem = list_grs[idx_elem] 
                        gt_elem = list_gts[idx_elem] 
                        region_mask_elem = list_region_mask[idx_elem] 
                        
                        nb_annotated_patches_extracted = 0
                        while nb_annotated_patches_extracted < nb_patches:
                            gr_chunks_arr, gt_chunks_arr = getRandomSamples_Given_Image(gr_elem, gt_elem, region_mask_elem, min(batch_size, nb_patches-nb_annotated_patches_extracted), window_shape[0], window_shape[1], augmentation_types)
                            nb_annotated_patches_extracted += len(gr_chunks_arr)
                            yield gr_chunks_arr, gt_chunks_arr
                
                
                
                '''
                while (True):
                    for page in data_pages:   
                        if utilConst.AUGMENTATION_RANDOM in augmentation_types:
                            assert(nb_patches != -1)

                            gr_chunks_arr, gt_chunks_arr = getRandomSamples(page, adapt_size, scaleFactor, min(batch_size, nb_patches), nb_annotated_patches, window_shape[0], window_shape[1], augmentation_types, vertical_reduction_regions, considered_classes)
                            if gr_chunks_arr is None and gt_chunks_arr is None:
                                idx_tries+=1
                                if idx_tries > len(data_pages):
                                    error_msg = 'It is not possible to annotate samples'
                                    print(error_msg)
                                    raise Exception(error_msg)
                                continue
                            else:
                                idx_tries=0
                                yield gr_chunks_arr, gt_chunks_arr
                        else:
                            assert(nb_annotated_patches == nb_patches)
                            real_patches = get_number_annotated_patches(page, window_shape[0], window_shape[1], nb_annotated_patches, considered_classes)
                            if nb_annotated_patches == -1:
                                nb_annotated_patches_real = real_patches
                                np_patches_real = real_patches
                            else:
                                nb_annotated_patches_real = nb_annotated_patches
                                np_patches_real = nb_patches
                            gr, gt, regions_mask, n_annotated_patches_real = get_image_with_gt(page[0], page[1], nb_annotated_patches_real, vertical_reduction_regions, True, considered_classes, using_mask)
                            idx_patch = 0
                            while idx_patch < min(n_annotated_patches_real, nb_annotated_patches_real):    
                                samples = getSequentialSamples(gr, gt, regions_mask, idx_patch, min(batch_size, real_patches), n_annotated_patches_real, n_annotated_patches_real, window_shape[0], window_shape[1], augmentation_types)
                                if samples is not None:
                                    idx_patch += len(samples[0])
                                    yield samples[0], samples[1]
                                else:
                                    idx_patch = min(n_annotated_patches_real, nb_annotated_patches)
                '''

def __run_validations(pred, gt):
    assert(isinstance(pred, np.ndarray))
    assert(isinstance(gt, np.ndarray))

    assert(np.issubdtype(pred.dtype.type, np.bool_))
    assert(np.issubdtype(gt.dtype.type, np.bool_))

    assert(len(pred) == len(gt))
    assert(pred.shape[0]==gt.shape[0])


def __calculate_metrics(prediction, gt):
    __run_validations(prediction, gt)

    not_prediction = np.logical_not(prediction)
    not_gt = np.logical_not(gt)

    tp = np.logical_and(prediction, gt)
    tn = np.logical_and(not_prediction, not_gt)
    fp = np.logical_and(prediction, not_gt)
    fn = np.logical_and(not_prediction, gt)

    tp = (tp.astype('int32')).sum()
    tn = (tn.astype('int32')).sum()
    fp = (fp.astype('int32')).sum()
    fn = (fn.astype('int32')).sum()

    epsilon = 0.00001
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    fm = 2 * (precision * recall) / (precision + recall + epsilon)
    specificity = tn / (tn + fp + epsilon)

    gt = gt.astype('int32')
    prediction = prediction.astype('int32')

    difference = np.absolute(prediction - gt)
    totalSize = np.prod(gt.shape)
    error = float(difference.sum()) / float(totalSize)

    return {'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn,
            'error':error, 'accuracy':accuracy, 'precision':precision,
            'recall':recall, 'fm':fm, 'specificity':specificity}



#imutils version adapted to RGB
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    if len(image.shape) >= 3:
        image_out = np.zeros([nH, nW,image.shape[2]])
        for channel in range(image.shape[2]):
            
            image_rotated = cv2.warpAffine(image[:,:,channel], M, (nW, nH))
            image_out[:,:,channel] = image_rotated        
    else:
        image_out = cv2.warpAffine(image, M, (nW, nH))
    
    return image_out

#https://stackoverflow.com/questions/69050464/zoom-into-image-with-opencv
def zoom_at(img, zoom, coord=None):
    """
    Simple image zooming without boundary checking.
    Centered at "coord", if given, else the image center.

    img: numpy.ndarray of shape (h,w,:)
    zoom: float
    coord: (float, float)
    """
    # Translate to zoomed coordinates
    h, w, _ = [ zoom * i for i in img.shape ]
    
    if coord is None: cx, cy = w/2, h/2
    else: cx, cy = [ zoom*c for c in coord ]
    
    img = cv2.resize( img, (0, 0), fx=zoom, fy=zoom)
    img = img[ int(round(cy - h/zoom * .5)) : int(round(cy + h/zoom * .5)),
               int(round(cx - w/zoom * .5)) : int(round(cx + w/zoom * .5)),
               : ]
    
    return img

def apply_augmentation(x_image, y_image, regions_mask, type_augmentation, value_augmentation=None):
    x_image_out = None
    y_image_out = None
    type_augmentation_out = None
    
    if y_image is None:
        y_image_float = None
    else:
        y_image_float = y_image.astype(np.float64)
    regions_mask_float = regions_mask.astype(np.float64)
    regions_mask_out = regions_mask_float 
    
    if type_augmentation == utilConst.AUGMENTATION_NONE:
        x_image_out = x_image
        y_image_out = y_image
        regions_mask_out = regions_mask
        type_augmentation_out = (type_augmentation, 0)
    elif type_augmentation == utilConst.AUGMENTATION_FLIPH:
        x_image_out = cv2.flip(x_image*1.0, 1)
        if y_image is not None:
            y_image_out = cv2.flip(y_image*1.0, 1)
        regions_mask_out = cv2.flip(regions_mask*1.0, 1)
        type_augmentation_out = (type_augmentation, 1)
        
    elif type_augmentation == utilConst.AUGMENTATION_FLIPV:
        x_image_out = cv2.flip(x_image*1.0, -1)
        if y_image is not None:
            y_image_out = cv2.flip(y_image*1.0, -1)
        regions_mask_out = cv2.flip(regions_mask*1.0, -1)
        type_augmentation_out = (type_augmentation, -1)
    elif type_augmentation == utilConst.AUGMENTATION_ROTATION:
        if value_augmentation is None:
            angle = random.uniform(-3, 3)
        else:
            angle = value_augmentation
        x_image_out = rotate_bound(x_image, angle)
        if y_image is not None:
            y_image_out = rotate_bound(y_image_float, angle)
        regions_mask_out = (rotate_bound(regions_mask_float, angle) > 0) * 1
        if y_image is not None:
            y_image_out = apply_mask(y_image_out, regions_mask_out)

        type_augmentation_out = (type_augmentation, angle)
        
    elif type_augmentation == utilConst.AUGMENTATION_SCALE:
        if value_augmentation is None:
            zoom_factor = random.uniform(0.95, 1.05)
        else:
            zoom_factor = value_augmentation
        ROWS = x_image.shape[0]
        COLS = x_image.shape[1]

        x_image_out = cv2.resize(x_image, None, fx=zoom_factor, fy=zoom_factor)
        if y_image is not None:
            y_image_out = cv2.resize(y_image_float, None, fx=zoom_factor, fy=zoom_factor)
        regions_mask_out = cv2.resize(regions_mask_float, None, fx=zoom_factor, fy=zoom_factor)
        if y_image is not None:
            y_image_out = apply_mask(y_image_out, regions_mask_out)
        type_augmentation_out = (type_augmentation, zoom_factor)

        
    elif type_augmentation == utilConst.AUGMENTATION_DROPOUT:
        assert (False)
        
    regions_mask_out = (regions_mask_out>0.5)*1
    l = np.where((regions_mask_out == 0))
    x_image_out[l] = utilConst.kPIXEL_VALUE_FOR_MASKING
    y_image_out[l] = 0
                
    return x_image_out, y_image_out, regions_mask_out, type_augmentation_out

#------------------------------------------------------------------------------
def run_test(y_pred, y_gt, threshold=.5):
    prediction = y_pred.copy()
    gt = y_gt.copy()

    if threshold is not None:
        prediction = (prediction > threshold)
    else:
        prediction = (prediction > 0.5)

    gt = gt > 0.5

    r = __calculate_metrics(prediction, gt)

    return r



def get_best_threshold(y_pred, y_test, verbose=1, args_th=None):
    best_fm = -1
    best_th = -1
    prec = 0.
    recall = 0.
    
    if args_th is None:
        for i in range(1, 10, 1):
            th = float(i) / 10.0
            #print('Threshold:', th)
            results = run_test(y_pred, y_test, threshold=th)
            fm = results['fm']
            if fm > best_fm:
                best_fm = fm
                best_th = th
                prec = results['precision']
                recall = results['recall']
        if verbose:
            print('Best threshold:', best_th)
            print("Best Fm: %.4f " % best_fm, 
                    "P: %.3f " % prec,
                    "R: %.3f " % recall)
    else:
        results = run_test(y_pred, y_test, threshold=args_th)
        best_fm = results['fm']
        best_th = args_th
        prec = results['precision']
        recall = results['recall']
        if verbose:
            print('Threshold:', best_th)
            print("Fm: %.4f " % best_fm, 
                    "P: %.3f " % prec,
                    "R: %.3f " % recall)

    return best_fm, best_th, prec, recall


def extract_annotated_samples_and_region_mask(path_model, train_data, window_shape, vertical_reduction_regions, nb_annotated_patches=-1, considered_classes=["staff", "empty-staff"], with_masked_input=True):
    
    window_w = window_shape[0]
    window_h = window_shape[1]
    
    idx = 0

    for page_train in train_data:
        page_src = page_train[0]
        page_gt = page_train[1]

        idx+=1
        print("Processing train data..." + str(idx) + "/" + str(len(train_data)) + ": " + page_src)
        
        gr, gt, region_mask, n_annotated_patches_real = get_image_with_gt(page_src, page_gt, nb_annotated_patches, 0., True, considered_classes, with_masked_input)
        gt=gt>0.5
        
        path_result = path_model.replace("models/modelCNN/", "train/").replace(".h5", "/") + page_train[0].replace("datasets/", "")
        X_samples, Y_samples = extract_list_annotated_samples(page_src, page_gt, nb_annotated_patches, window_w, window_h, vertical_reduction_regions, True, considered_classes)
        assert(len(X_samples) == len(Y_samples))
        utilIO.saveImage((gt)*255, path_result + "_gt.png") 
        utilIO.saveImage((gr)*255, path_result + "_gr.png")
        utilIO.saveImage((region_mask)*255, path_result + "_annotated_regions.png")

        for idx_sample in range(len(X_samples)):
            X_sample = X_samples[idx_sample]
            Y_sample = Y_samples[idx_sample]
            utilIO.saveImage((1-X_sample)*255, path_result + "/samples/" + str(idx_sample) + "_gr.png") 
            utilIO.saveImage((Y_sample)*255, path_result + "/samples/" + str(idx_sample) + "_gt.png") 
    pass

def getBoundingBoxes(image, vertical_expansion_regions=0., val=100):
    threshold = val
    minContourSize= int(image.shape[0]*image.shape[1]*0.0025)
    
    img = np.copy(image)
    ROWS = img.shape[0]
    COLS = img.shape[1]

    for j in range(COLS):
        img[0, j] = 0
        img[1, j] = 0
        img[2, j] = 0

        img[ROWS-1, j] = 0
        img[ROWS-2, j] = 0
        img[ROWS-3, j] = 0
    
    for i in range(ROWS):
        img[i, 0] = 0
        img[i, 1] = 0
        img[i, 2] = 0

        img[i, COLS-1] = 0
        img[i, COLS-2] = 0
        img[i, COLS-3] = 0
    
    
    
    #minContourSize = 500
    im = np.uint8(img)
    kernel = np.ones((3,3),np.uint8)
    im = cv2.erode(im,kernel,iterations = 5)
    im = cv2.dilate(im,kernel,iterations = 5)

    canny_output = cv2.Canny(im, threshold, threshold * 2)
    
    contours, herarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create hull array for convex hull points
    hull = []
    boundRect = []

    # calculate points for each contour
    for i in range(len(contours)):

        contour_i = contours[i]
        contour_poly = cv2.approxPolyDP(contour_i, 3, True)
        hull_i = cv2.convexHull(contour_poly, False)

        area = cv2.contourArea(hull_i)

        if (area > minContourSize):
            bbox_i = cv2.boundingRect(hull_i)

            height = bbox_i[3]
            vertical_dilation = (height * vertical_expansion_regions) // 2
            
            
            rect_by_corners = (max(0,bbox_i[1]-vertical_dilation), bbox_i[0], min(im.shape[0],bbox_i[1]+bbox_i[3]+vertical_dilation), bbox_i[0]+bbox_i[2])
            
            if (bbox_i[3] > bbox_i[2]):# If tt is a vertical region, we ignore it.
                continue

            if (bbox_i[3] / bbox_i[2] < 0.05):# If width and height are so much different, we ignore the region
                continue 
             
            boundRect.append(rect_by_corners)

        #if (cv.contourArea(hull_i) > 0):
            

    return boundRect

def compute_Fscore_from_TP_FP_FN(tp, fp, fn):
    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    
    if (precision+recall) == 0:
        epsilon = 0.00000001
    else:
        epsilon = 0.
    return 2 * (precision*recall) / (precision+recall + epsilon), precision, recall



def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def calculateAveragePrecisionLists(list_gt_bbox, list_pred_bbox, ths):
    recall_list = [] 
    precision_list = []
    for th in ths:
        tp_th = 0
        fp_th = 0
        fn_th = 0
        for idx in range(len(list_gt_bbox)):
            gt_boxes = list_gt_bbox[idx]
            pred_boxes = list_pred_bbox[idx]
            fscore, precision, recall, all_overlapping_area_page, tp, fn, fp, list_overlapping_area_page, num_regions_page, overlapping_area_tp_page, list_ordered_overlapping_page, list_matched_true_positives_page = FscoreRegion.getFscoreRegions(gt_boxes, pred_boxes, th=th)
            
            tp_th += tp
            fp_th += fp
            fn_th += fn

        f1, precision, recall = FscoreRegion.getF1PrecisionRecall(tp_th, fp_th, fn_th)
        recall_list.append(recall)
        precision_list.append(precision)

    mAP, precision_curve, recall_curve = compute_ap(recall_list, precision_list)
    return mAP


def draw_BoundingBoxes(image, bounding_boxes, border=2):
    image_with_bboxes = np.copy(image)
    for predicted_bbox in bounding_boxes:
        x1 = int(predicted_bbox[0])
        y1 = int(predicted_bbox[1])
        x2 = int(predicted_bbox[2])
        y2 = int(predicted_bbox[3])
        image_with_bboxes[x1:x2, y1:y2] = image_with_bboxes[x1:x2, y1:y2] // 2
        image_with_bboxes[x1:x2, y1:y2] += [110,40,40]

        border = 2
        image_with_bboxes[x1:x2, y1-border:y1+border] = [110,5,5]
        image_with_bboxes[x1:x2, y2-border:y2+border] = [110,5,5]
        image_with_bboxes[x1-border:x1+border, y1:y2] = [110,5,5]
        image_with_bboxes[x2-border:x2+border, y1:y2] = [110,5,5]
    return image_with_bboxes

def compute_best_threshold(path_model, val_data, adapt_size, scaleFactor, standard, batch_size, window_shape, th_iou, considered_classes=["staff", "empty-staff"], nb_annotated_patches=-1, threshold=None, with_masked_input=True, istest=True, vertical_reduction_regions=0., save_images=False):
    model = tf.keras.models.load_model(path_model)
    window_w = window_shape[0]
    window_h = window_shape[1]
    speed_factor = 3

    predictions = []
    gts = []
    list_region_mask = []
    list_coords_with_annotations = []
    idx = 0
    dict_predictions = {}
    for page_test in val_data:
        page_src = page_test[0]
        page_gt = page_test[1]

        idx+=1
        print("Processing..." + str(idx) + "/" + str(len(val_data)) + ": " + page_src)
        if istest:
            gr_orig, gt_orig, region_mask_orig, n_annotated_patches_real = get_image_with_gt(page_src, page_gt, -1, 0., False, considered_classes, False)
        else:
            gr_orig, gt_orig, region_mask_orig, n_annotated_patches_real = get_image_with_gt(page_src, page_gt, nb_annotated_patches, 0., True, considered_classes, True)
            
        coords_masked_original_size = np.where((region_mask_orig == 0))
        
        if not standard and adapt_size:
            gr = image_resize_given_ratio(gr_orig, ratio_width = None, ratio_height = scaleFactor)
            region_mask = image_resize_given_ratio(region_mask_orig, ratio_width = None, ratio_height = scaleFactor)
            region_mask = region_mask>0.5
        else:
            if standard:
                gr = cv2.resize(gr_orig*1.0, (window_shape[1], window_shape[0]))
                region_mask = cv2.resize(region_mask_orig*1.0, (window_shape[1], window_shape[0]))
            else:
                gr = gr_orig
                region_mask = region_mask_orig
        gt_orig=gt_orig>0.5
        l = np.where((region_mask == 0))
        gr[l] = 0

        if standard:
            gr_to_predict = resizeImage(gr, (window_w, window_h, 3))
        else:
            gr_to_predict = gr

        window_shape_speeded = (window_shape[0]*speed_factor, window_shape[1]*speed_factor)
        prediction = predict_image(model, gr_to_predict, -1, window_shape_speeded)
        if standard:
            prediction = cv2.resize(prediction*1.0, (gr_orig.shape[1], gr_orig.shape[0]))  #resizeBack(prediction, (gr_orig.shape[0], gr_orig.shape[1]))
        
        if adapt_size:
            prediction = cv2.resize(prediction*1.0, (gr_orig.shape[1],gr_orig.shape[0]))
            #prediction = image_resize_given_ratio(prediction, ratio_width = None, ratio_height = 1/scaleFactor)
            
        coords_with_annotations = np.where((region_mask.flatten())!=0)

        prediction[coords_masked_original_size] = 0
        
        dict_predictions[page_src]  = prediction
        
        predictions.append(prediction)
        gts.append(gt_orig)
        list_region_mask.append(region_mask_orig)
        list_coords_with_annotations.append(coords_with_annotations)

        if istest:
            subfolder="test/"
        else:
            subfolder="train/"
            
        if save_images:
            path_result = path_model.replace("models/modelCNN/", subfolder).replace(".h5", "/") + page_test[0].replace("datasets/", "")
            utilIO.saveImage((gt_orig)*255, path_result + "_gt.png") 
            utilIO.saveImage((prediction)*255, path_result + "_pred.png")
            utilIO.saveImage((gr_orig)*255, path_result + "_gr.png")
            utilIO.saveImage((region_mask)*255, path_result + "_annotated_regions.png")

            if istest and threshold is not None:
                utilIO.saveImage((prediction>threshold)*255, path_result + "_th_pred.png")

                predicted_bboxes = getBoundingBoxes((prediction>threshold)*255, vertical_reduction_regions)
                gr_with_bboxes = draw_BoundingBoxes((1-gr_orig)*255, predicted_bboxes, border=2)                
                utilIO.saveImage((gr_with_bboxes), path_result + "_gr_with_bboxes.png")
        

    best_fm = 0
    best_th = 0
    best_prec = 0
    best_recall = 0
    best_tp = 0
    best_fp = 0
    best_fn = 0
    best_iou_avg = 0
    best_mAP = 0
    if threshold is None:
        list_th_bin = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        list_th_bin = [threshold]
        

    was_improved = False        
    list_Fscore = []
    list_prec = []
    list_recall = []
    list_tp_total = []
    list_fp_total = []
    list_fn_total = []
    list_iou_avg_th = []
    list_mAP_th = []
    list_ordered_overlapping_all_pages_th = []
    
            
    
    for th_bin in list_th_bin:
        list_discard_th_bin = []
        list_mAP_page = []
        tp_total = 0
        fn_total = 0
        fp_total = 0
        list_overlaping = []
        list_bboxes_gts = []
        list_bboxes_predictions = []
        list_ordered_overlapping_all_pages = []
        idx_page = 0
        mAP = 0
        for prediction in predictions:
            page_gr = val_data[idx_page][0]
            page_gt = val_data[idx_page][1]
            region_mask = list_region_mask[idx_page]
            idx_page+=1

            coords_region_used = np.where((region_mask == 1))

            predicted_bboxes = getBoundingBoxes((prediction>th_bin)*255, vertical_reduction_regions)
            if istest:
                gt_bboxes = utilIO.get_List_boundingBoxes_from_JSON(page_gt, -1, considered_classes)
                discard_th_bin = False
            else:
                gt_bboxes = utilIO.get_List_boundingBoxes_from_JSON(page_gt, nb_annotated_patches, considered_classes)
                discard_th_bin = not (np.amin(prediction[coords_region_used]>th_bin) == np.amax(prediction[coords_region_used]>th_bin))

            list_discard_th_bin.append(discard_th_bin)
            if len(gt_bboxes) == 0:
                continue
            fscore_page, precision_page, recall_page, all_overlapping_area_page, tp, fn, fp, list_overlapping_area_page, num_regions_page, overlapping_area_tp_page, list_ordered_overlapping_page, list_matched_true_positives_page = FscoreRegion.getFscoreRegions(gt_bboxes, predicted_bboxes, th=th_iou)
            list_bboxes_gts.append(gt_bboxes)
            list_bboxes_predictions.append(predicted_bboxes)
            list_ordered_overlapping_all_pages.append(list_matched_true_positives_page)
            if tp is not None:
                tp_total += tp
                fn_total += fn
                fp_total += fp
                for overlaping_area_page in list_overlapping_area_page:
                    list_overlaping.append(overlaping_area_page)
            
        ths_IoU = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        mAP = calculateAveragePrecisionLists(list_bboxes_gts, list_bboxes_predictions, ths_IoU)
        
        iou_avg_th = np.average(list_overlaping)
        Fscore, prec, recall = compute_Fscore_from_TP_FP_FN(tp_total, fp_total, fn_total)
        
        list_Fscore.append(Fscore)
        list_prec.append(prec)
        list_recall.append(recall)
        list_tp_total.append(tp_total)
        list_fp_total.append(fp_total)
        list_fn_total.append(fn_total)
        list_iou_avg_th.append(iou_avg_th)
        list_mAP_th.append(mAP)
        list_ordered_overlapping_all_pages_th.append(list_ordered_overlapping_all_pages)
        
        discard_th_bin = all(list_discard_th_bin)
        if not discard_th_bin and iou_avg_th > best_iou_avg:
            best_fm = Fscore
            best_th = th_bin
            best_prec = prec
            best_recall = recall
            best_tp = tp_total
            best_fp = fp_total
            best_fn = fn_total
            best_iou_avg = iou_avg_th
            best_mAP = mAP
            best_list_ordered_overlapping_all_pages = list_ordered_overlapping_all_pages
            was_improved = True
        
                
    if not was_improved:
        best_fm = list_Fscore[0]
        best_th = list_th_bin[0]
        best_prec = list_prec[0]
        best_recall = list_recall[0]
        best_tp = list_tp_total[0]
        best_fp = list_fp_total[0]
        best_fn = list_fn_total[0]
        best_iou_avg = list_iou_avg_th[0]
        best_mAP = list_mAP_th[0]
        best_list_ordered_overlapping_all_pages = list_ordered_overlapping_all_pages_th[0]

    if istest and save_images:
        extract_Regions_and_save(best_list_ordered_overlapping_all_pages, list_bboxes_gts, val_data, path_model, considered_classes, th_iou)
    
    
    #predictions = np.array(predictions)
    #gts = np.array(gts)
    #best_fm, best_th, prec, recall = get_best_threshold(predictions, gts, verbose=1, args_th=threshold)
    
    

    return best_fm, best_th, best_prec, best_recall, best_tp, best_fp, best_fn, best_iou_avg, best_mAP, dict_predictions


def obtainSequence(gt_bboxes, gt_bbox):
    for gt_bbox_item in gt_bboxes:
        if gt_bbox == gt_bbox_item[0]:
            return gt_bbox_item[1]
    return None

def extract_Regions_and_save(best_list_ordered_overlapping_all_pages, list_bboxes_gts, val_data, path_model, considered_classes, threshold):
    idx_page = 0
    for page_test in val_data:
        page_src = page_test[0]
        page_gt = page_test[1]
        
        list_bboxes_gt_page = list_bboxes_gts[idx_page]
        list_ordered_overlapping_page = best_list_ordered_overlapping_all_pages[idx_page]
        
        path_result = path_model.replace("models/modelCNN/", "regions/th_iou_"+str(threshold)+"/").replace(".h5", "/") + page_test[0].replace("datasets/", "")
        
        gr_orig, gt_orig, region_mask_orig, n_annotated_patches_real = get_image_with_gt(page_src, page_gt, -1, 0., False, considered_classes, False)
        
        gt_bboxes = utilIO.get_List_boundingBoxes_from_JSON_with_sequences(page_gt, -1, considered_classes)
        idx_region = 0
        for matched_bboxes in list_ordered_overlapping_page:
            gt_bbox = matched_bboxes.gt_bbox
            pred_bbox = matched_bboxes.pred_bbox
            overlapped_area = matched_bboxes.overlapped_area

            list_tokens_sequence = obtainSequence(gt_bboxes, gt_bbox)
            
            if len(list_tokens_sequence) > 0:
            
                gr_region = gr_orig[int(pred_bbox[0]):int(pred_bbox[2]), int(pred_bbox[1]):int(pred_bbox[3])]
                gt_region = gr_orig[int(gt_bbox[0]):int(gt_bbox[2]), int(gt_bbox[1]):int(gt_bbox[3])]
                path_result_GT = path_result.replace("SRC/", "GT/")
                path_result_PRED = path_result.replace("SRC/", "PRED/")
                
                utilIO.saveImage((1-gr_region)*255, path_result_PRED + "_page_" + str(idx_page) + "_idx_" + str(idx_region) + "_pred.png")
                utilIO.saveImage((1-gt_region)*255, path_result_GT + "_page_" + str(idx_page) + "_idx_" + str(idx_region) + "_gt.png")
                sequence_txt = "\t".join(list_tokens_sequence)
                utilIO.writeString(sequence_txt, path_result_GT + "_page_" + str(idx_page) + "_idx_" + str(idx_region) + "_gt.png.agnostic")
                utilIO.writeString(sequence_txt, path_result_PRED + "_page_" + str(idx_page) + "_idx_" + str(idx_region) + "_pred.png.agnostic")
            idx_region+=1
        idx_page+=1

def compute_metrics(config, path_model, test_data, batch_size, window_shape, th_iou, considered_classes=["staff", "empty-staff"], threshold=None, with_masked_input=True):
    import CNNmodel
    no_mask = not with_masked_input
    model = CNNmodel.get_model(window_shape, no_mask, config.n_la, config.nb_fil, config.ker, dropout=config.drop, stride=2)
    model.load_weights(path_model)
    
    #model = tf.keras.models.load_model(path_model)
    window_w = window_shape[0]
    window_h = window_shape[1]
 
    
    idx = 0
    dict_predictions = {}
    for page_test in test_data:
        page_src = page_test[0]
        page_gt = page_test[1]

        idx+=1
        print("Processing..." + str(idx) + "/" + str(len(test_data)) + ": " + page_src)
        
        gr, gt, _, _ = get_image_with_gt(page_src, page_gt, -1, 0., True, considered_classes, False)
        _, _, regions_mask, _ = get_image_with_gt(page_src, page_gt, config.n_an, 0., True, considered_classes, False)
        gt=gt>0.5
        prediction_matrix = predict_image(model, gr, -1, window_shape)

        
        path_result = path_model.replace("models/modelCNN/", "tests/").replace(".h5", "/") + page_test[0].replace("datasets/", "")
        utilIO.saveImage((gt)*255, path_result + "_gt.png") 
        utilIO.saveImage((gr)*255, path_result + "_gr.png")
        utilIO.saveImage((prediction_matrix)*255, path_result + "_pred.png")
        utilIO.saveImage((prediction_matrix>threshold)*255, path_result + "_pred_th.png")
        utilIO.saveImage((regions_mask)*255, path_result + "_annotated_regions.png")
        
        
        gr=None
        gt=None

        if utilConst.KEY_RESULT not in dict_predictions:
            dict_predictions[utilConst.KEY_RESULT] = {}    
        dict_predictions[utilConst.KEY_RESULT][page_src] = {}
        dict_predictions[utilConst.KEY_RESULT][page_src][0] = prediction_matrix

    dict_results = {}
    
    predictions = np.array(list())
    gts = np.array(list())
    for page_test in test_data:
        page_src = page_test[0]
        page_gt = page_test[1]
        
        gr, gt, _, _ = get_image_with_gt(page_src, page_gt, -1, 0., True, considered_classes, with_masked_input)
        coords_with_annotations = np.where((dict_predictions[utilConst.KEY_RESULT][page_src][0].flatten())!=utilConst.kPIXEL_VALUE_FOR_MASKING)
        predictions = np.concatenate((predictions, (dict_predictions[utilConst.KEY_RESULT][page_src][0].flatten())[coords_with_annotations]))
        gts = np.concatenate((gts, (gt.flatten())[coords_with_annotations]))
    if len(predictions) != 0 and len(gts) != 0:
        best_fm, best_th, prec, recall = get_best_threshold(predictions, gts, verbose=1, args_th=threshold)
        if utilConst.KEY_RESULT not in dict_results:
            dict_results[utilConst.KEY_RESULT] = {}    
        dict_results[utilConst.KEY_RESULT][0] = (best_fm, prec, recall)


    return dict_results, dict_predictions


def isMultipleTimes(value, multiple, times):
    while value % multiple == 0:
        value  = value / multiple
        times -= 1
    if times == 0:
        return True
    return False
def findNextValueMultipleTimes(value, multiple, times):
    while isMultipleTimes(value, multiple, times) == False:
        value += 1
    return value


def predict_image(model, gr_norm, nb_sequential_patches, window_shape):
    
    window_w = window_shape[0]
    window_h = window_shape[1]
        
    ROWS = gr_norm.shape[0]
    COLS = gr_norm.shape[1]

    #prediction = np.ones((ROWS, COLS))*(-1)
    prediction = np.zeros((ROWS, COLS))
    margin = 10
    patch_counter = 0



    try:
        list_patches_batch=[]
        list_patches_batch.append(gr_norm)
        patch_gr_arr = np.array(list_patches_batch)

        prediction_model = model.predict(patch_gr_arr, verbose=0)[0,:,:,0]
        prediction = prediction_model
        return prediction
    except:
        try:
            print ("Modifying resolution...")
            topmargin = abs(ROWS-findNextValueMultipleTimes(ROWS, 2, 4))
            leftmargin = abs(COLS -findNextValueMultipleTimes(COLS, 2, 4))
            p = np.zeros((ROWS+topmargin, COLS+leftmargin, 3))
            p[topmargin:, leftmargin:, :] = gr_norm[:,:,:]
            list_patches_batch=[]
            list_patches_batch.append(p)
            patch_gr_arr = np.array(list_patches_batch)
            prediction_model = model.predict(patch_gr_arr, verbose=0)[0,:,:,0]
            prediction[:, :] = prediction_model[topmargin:, leftmargin:]

            return prediction
        except:
   
            for row in range(window_w//2, ROWS+window_w//2-1, window_w//2):
                for col in range(window_h//2, COLS+window_h//2-1, window_h//2):
                    row = min(row, ROWS-window_w//2)
                    col = min(col, COLS-window_h//2)
                    
                    patch_gr = gr_norm[row-window_w//2:row-window_w//2+window_w, col-window_h//2:col-window_h//2+window_h]
                    patch_gr_arr = np.array(patch_gr)
                    patch_gr_arr = np.reshape(patch_gr_arr, (1, patch_gr_arr.shape[0], patch_gr_arr.shape[1], patch_gr_arr.shape[2]))
                    
                    predicted_patch = model.predict(patch_gr_arr, verbose=0)[0,:,:,0]
                    
                    prediction[row-window_w//2+margin:row-window_w//2+window_w-margin, col-window_h//2+margin:col-window_h//2+window_h-margin] = np.maximum(prediction[row-window_w//2+margin:row-window_w//2+window_w-margin, col-window_h//2+margin:col-window_h//2+window_h-margin], predicted_patch[margin:-margin,margin:-margin])
                    patch_counter+=1
                    if (nb_sequential_patches != -1 and patch_counter >=nb_sequential_patches*2) or nb_sequential_patches == 1:
                        return prediction

    return prediction


def test_model(config, path_model, test_data, window_shape, threshold, with_masked_input, th_iou, considered_classes):
    dict_results, dict_predictions = compute_metrics(config=config, path_model=path_model, test_data=test_data, batch_size=1, window_shape=window_shape, threshold=threshold, with_masked_input=with_masked_input, th_iou=th_iou, considered_classes=considered_classes)
    
    pathfolder_result = path_model.replace(".h5", "/").replace("models/", "results/")
    pathfolder_result_bin = path_model.replace(".h5", "/").replace("models/", "results/bin/")

        
    return dict_results

    
    
    
