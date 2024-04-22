# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os, warnings


gpu = sys.argv[ sys.argv.index('-gpu') + 1 ] if '-gpu' in sys.argv else '0'
os.environ['PYTHONHASHSEED'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES']=gpu
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow CUDA load statements
#warnings.filterwarnings('ignore')

from keras import backend as K
import tensorflow as tf

import copy
import argparse
import numpy as np


gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", gpus)

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs

    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


import utilArgparse
import utilConst
import utilIO
import util
import CNNmodel

#util.init()

K.set_image_data_format('channels_last')




# ----------------------------------------------------------------------------
def menu():
    parser = argparse.ArgumentParser(description='Binarization with masking layer and oversampling')

    
    parser.add_argument('-path_model', required=False, help='Path to the model. If it is not provided, it is automatically calculated.')
    
    parser.add_argument('-db_train_txt', required=True, help='Path to the txt file for training')
    parser.add_argument('-db_val_txt', required=True, help='Path to the txt file for validation')
    parser.add_argument('-db_test_txt', required=False, help='Path to the txt file for testing')

    parser.add_argument('-aug',   nargs='*',
                        choices=utilConst.AUGMENTATION_CHOICES,
                        default=[utilConst.AUGMENTATION_NONE], 
                        help='Data augmentation modes')

    parser.add_argument('-cls',   nargs='*',
                        choices=utilConst.CLASS_CHOICES,
                        default=[utilConst.CLASS_STAFF, utilConst.CLASS_EMPTY_STAFF], 
                        help='Classes to be considered.')
    
    parser.add_argument('-npatches', default=-1, dest='n_pa', type=int,   help='Number of patches to be extracted from training data')
    
    parser.add_argument('-n_annotated_patches', default=-1, dest='n_an', type=int,   help='Number of patches to be extracted from training data')

    parser.add_argument('-window_w', default=256, dest='win_w', type=int,   help='width of window')
    parser.add_argument('-window_h', default=256, dest='win_h', type=int,   help='height of window')

    parser.add_argument('-l',          default=4,        dest='n_la',     type=int,   help='Number of layers')
    parser.add_argument('-f',          default=64,      dest='nb_fil',   type=int,   help='Number of filters')
    parser.add_argument('-k',          default=5,        dest='ker',            type=int,   help='kernel size')
    parser.add_argument('-drop',   default=0.2,        dest='drop',          type=float, help='dropout value')
    parser.add_argument('-iou',   default=0.55,        dest='iou',          type=float, help='Threshold of IoU for calculating F-score metrics')
    
    parser.add_argument('-vr',   default=0.,        dest='vr',          type=float, help='vertical reduction for bounding boxes used in training')
    
   

    parser.add_argument('-pages_train',   default=-1,      type=int,   help='Number of pages to be used for training. -1 to load all the training set.')

    parser.add_argument('-e',           default=200,    dest='ep',            type=int,   help='nb_epoch')
    parser.add_argument('-b',           default=16,     dest='ba',               type=int,   help='batch size')
    
    parser.add_argument('--test',   action='store_true', help='Only run test')
    
    
    parser.add_argument('-res', required=False, help='File where append the results.')
    parser.add_argument('-gpu',    default='0',    type=str,   help='GPU')
    parser.add_argument('-no_mask', required=False, action='store_true', help='File where append the results.')

    parser.add_argument('-standard', required=False, action='store_true', help='Standard mode, resizing the input image to a specific.')
    parser.add_argument('-adapt-size-patch', required=False, action='store_true', dest='adapt_size', help='Activate the adaptation of input size according to the size of the windows. Not used for standard mode.')
    parser.add_argument('-save', required=False, action='store_true', dest='save', help='Saving resulting images with the predictions.')
    

    args = parser.parse_args()

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args

def tpc_result(result):
    return round(result*100,1)
  
def number_to_string(number):
    return str(tpc_result(number)).replace(".",",")



if __name__ == "__main__":
    config = menu()
    print (config)
    if config.path_model is None:
        path_model = utilIO.getPathModel(config)
    else:
        path_model = config.path_model
    #path_model = "models/modelCNN/modelCNN_aug__random__rot__scale___ba_32__cls__st__emp_st___dtr_txt_dbs__b-59-850_fd0_train.txt__db_val_txt_dbs__b-59-850_fd0_val.txt__drop_0.2__ep_200__ker_3__n_an_-1__n_la_3__n_pa_-1__nb_fil_32__nmsk__pt_-1__std__vr_0.4__win_h_512__win_w_512.h5"
    #path_model = "models/modelCNN/modelCNN_aug__random__rot__scale___ba_32__cls__st__emp_st___dtr_txt_dbs__SEILS_fd0_train.txt__db_val_txt_dbs__SEILS_fd0_val.txt__drop_0.2__ep_200__ker_3__n_an_-1__n_la_3__n_pa_-1__nb_fil_32__nmsk__pt_-1__std__vr_0.4__win_h_512__win_w_512.h5"
    #path_model = "models/modelCNN/modelCNN_aug__random__rot__scale___ba_32__cls__st__emp_st___dtr_txt_dbs__MT_c_fd0_train.txt__db_val_txt_dbs__MT_c_fd0_val.txt__drop_0.2__ep_200__ker_3__n_an_-1__n_la_3__n_pa_-1__nb_fil_32__nmsk__pt_-1__std__vr_0.4__win_h_512__win_w_512.h5"
    #path_model = "models/modelCNN/modelCNN_aug__random__rot__scale___ba_32__cls__st__emp_st___dtr_txt_dbs__MT_m_fd0_train.txt__db_val_txt_dbs__MT_m_fd0_val.txt__drop_0.2__ep_200__ker_3__n_an_-1__n_la_3__n_pa_-1__nb_fil_32__nmsk__pt_-1__std__vr_0.4__win_h_512__win_w_512.h5"
    #path_model = "models/modelCNN/modelCNN_aug__random__rot__scale___ba_32__cls__st__emp_st___dtr_txt_dbs__Patriarca_fd0_train.txt__db_val_txt_dbs__Patriarca_fd0_val.txt__drop_0.2__ep_200__ker_3__n_an_-1__n_la_3__n_pa_-1__nb_fil_32__nmsk__pt_-1__std__vr_0.4__win_h_512__win_w_512.h5"
    #path_model = "models/modenCNN/modelCNN_aug__random__rot__scale___ba_32__cls__st__emp_st___dtr_txt_dbs__Guatemala_fd0_train.txt__db_val_txt_dbs__Guatemala_fd0_val.txt__drop_0.2__ep_200__ker_3__n_an_-1__n_la_3__n_pa_-1__nb_fil_32__nmsk__pt_-1__std__vr_0.4__win_h_512__win_w_512.h5"
    print("Path model:")
    print(path_model)
    
    utilIO.createParentDirectory(path_model)
    
    input_shape = util.getInputShape(config)
    
    list_gt_train = utilIO.readListStringFile(config.db_train_txt)
    list_gt_val = utilIO.readListStringFile(config.db_val_txt)

    list_gt_test = None
    if config.db_test_txt is not None:
      list_gt_test = utilIO.readListStringFile(config.db_test_txt)

    list_src_train = utilIO.getList_SRCfiles_from_JSONfiles(list_gt_train)
    list_src_val = utilIO.getList_SRCfiles_from_JSONfiles(list_gt_val)
    list_src_test = utilIO.getList_SRCfiles_from_JSONfiles(list_gt_test)
    
    assert(len(list_src_train) == len(list_gt_train))

    train_data, val_data = util.create_Validation_and_Training_partitions(
                                        list_src_train=list_src_train, 
                                        list_gt_train=list_gt_train, 
                                        list_src_val=list_src_val, 
                                        list_gt_val=list_gt_val, 
                                        pages_train=config.pages_train)
    
    considered_classes = config.cls
    
    vertical_reduction_regions = config.vr
    th_iou = config.iou

    test_data = utilIO.match_SRC_GT_Images(list_src_test, list_gt_test)
    
    train_data = util.getListPages_with_content(train_data, considered_classes, config.n_an)
    val_data = util.getListPages_with_content(val_data, considered_classes, config.n_an)
    test_data = util.getListPages_with_content(test_data, considered_classes, -1)

    if config.test == False: # TRAINING MODE

      print("Training and validation partitioned...")
      print("\tTraining: %d" %(len(train_data)))
      print("\tValidation: %d" %(len(val_data)))

      augmentation_val = ["none"]
      if utilConst.AUGMENTATION_RANDOM in config.aug:
        augmentation_val = ["random"]
      
      model = CNNmodel.get_model((None, None, utilConst.kNUMBER_CHANNELS), config.no_mask, config.n_la, config.nb_fil, config.ker, dropout=config.drop, stride=2)
      
      num_pages_with_data = 0

      list_data_with_content = util.getListPages_with_content(train_data, considered_classes, config.n_an)
      number_train_pages_with_annotations = len(list_data_with_content)
      num_paches_per_image = config.n_pa // number_train_pages_with_annotations
    
      train_generator = util.create_generator_renewed(train_data, config.standard, config.adapt_size, config.no_mask, config.ba, input_shape, num_paches_per_image, config.n_an, config.aug, vertical_reduction_regions, considered_classes)
      val_generator = util.create_generator_renewed(val_data, True, False, config.no_mask, config.ba, input_shape, config.n_pa, config.n_an, augmentation_val, vertical_reduction_regions, considered_classes)
      
      nb_train_pages = len(train_data)
      nb_val_pages = len(val_data)
      
      epochs = config.ep
      patience = 20
      
      print("Number of effective epochs: " + str(epochs))
      print("Effective patience: " + str(patience))

      number_annotated_patches = util.get_number_annotated_patches(train_data, config.n_an, considered_classes)  
      number_annotated_patches_val = util.get_number_annotated_patches(val_data, config.n_an, considered_classes)  
      
      if config.standard:
          num_pages_with_data_train = util.get_number_pages_with_content(train_data, config.n_an, considered_classes)
          steps_per_epoch = int(np.ceil(num_pages_with_data / config.ba))

          num_pages_with_data_val = util.get_number_pages_with_content(val_data, config.n_an, considered_classes)
          steps_per_epoch_val = int(np.ceil(num_pages_with_data_val / config.ba))

          num_paches_per_image = config.n_pa
      else:
        if utilConst.AUGMENTATION_RANDOM in config.aug:
          assert(config.n_pa!=-1)

          steps_per_epoch = int(np.ceil((config.n_pa)/ config.ba))
          steps_per_epoch_val = int(np.ceil((config.n_pa)/ config.ba))
        else:
          print ("Number of annotated patches: " + str(number_annotated_patches))
          steps_per_epoch = np.ceil(number_annotated_patches/config.ba)
          steps_per_epoch_val = np.ceil(number_annotated_patches_val/config.ba)

      if number_annotated_patches > 0 and number_annotated_patches_val > 0:
        steps_per_epoch = max(1, steps_per_epoch)
        steps_per_epoch_val = max(1, steps_per_epoch_val)
        print("Steps per epoch: " + str(steps_per_epoch))
        print("Steps per epoch Val: " + str(steps_per_epoch_val))
        
        preepochs = 30
        
        num_pages_with_data_val = 0
        for page in val_data:
          gr, gt, regions_mask, n_annotated_patches_real = util.get_image_with_gt(page[0], page[1], config.n_an, config.vr, False, considered_classes, True)
          if n_annotated_patches_real > 0:
            num_pages_with_data_val +=1
        steps_per_epoch_val = num_pages_with_data_val
        if config.standard:
          steps_per_epoch *= 10
        CNNmodel.train(model, path_model, train_generator, val_generator, steps_per_epoch, steps_per_epoch_val, nb_val_pages, config.ba, preepochs=preepochs, epochs=epochs, patience=patience)
      else:
        print("No samples available with the ink rate considered. Train (" + str(number_annotated_patches) +") ; Val (" + str(number_annotated_patches_val) + ")")
        
    else: #TEST MODE
      
      #util.extract_annotated_samples_and_region_mask(path_model, train_data, input_shape, vertical_reduction_regions, config.n_an, considered_classes, with_masked_input=True)
      if config.adapt_size:
        scaleFactor = util.calculateScaleFactor_to_adapt_image_to_patches(train_data, input_shape[1], config.n_an, considered_classes)
        print("Scale factor to adapt to the windows: " + str(scaleFactor))
      else:
        scaleFactor = None
        
      number_annotated_patches = util.get_number_annotated_patches(train_data, config.n_an, considered_classes) 
      number_annotated_patches_val = util.get_number_annotated_patches(val_data, config.n_an, considered_classes)  

      max_number_annotated_patches = util.get_number_annotated_patches(train_data, -1, considered_classes) 
      max_number_annotated_patches_val = util.get_number_annotated_patches(val_data, -1, considered_classes)
      
      print("Obtaining best threshold...(Validation partition)")
      
      threshold=None
      
      if number_annotated_patches > 0 and number_annotated_patches_val > 0:
        print("Validation to optimize IoU...")
        
        best_fm_val, best_th_val, prec_val, recall_val, best_tp_val, best_fp_val, best_fn_val, best_iou_avg_val, best_mAP_val, dict_predictions = util.compute_best_threshold(path_model, val_data, config.adapt_size, scaleFactor, config.standard, config.ba, input_shape, th_iou, considered_classes, nb_annotated_patches=config.n_an, threshold=threshold, with_masked_input=True, istest=False, vertical_reduction_regions=vertical_reduction_regions, save_images=False)
        with_mask = not config.no_mask
        print("Testing...")
        best_fm_test, _, prec_test, recall_test, best_tp_test, best_fp_test, best_fn_test, best_iou_avg_test, best_mAP_test, dict_results = util.compute_best_threshold(path_model, test_data, config.adapt_size, scaleFactor, config.standard, config.ba, input_shape, th_iou, considered_classes, nb_annotated_patches=config.n_an, threshold=best_th_val, with_masked_input=False, istest=True, vertical_reduction_regions=vertical_reduction_regions, save_images=config.save)
        #dict_results = util.test_model(config, path_model, test_data, input_shape, best_th_val, with_mask, th_iou, considered_classes)
      else:
        print("No model is trained with this configuration...")
        best_fm_val = 0
        best_th_val = 0
        prec_val = 0
        recall_val = 0
        dict_predictions = None
        best_fm_test = 0
        prec_test = 0
        recall_test = 0
        best_iou_avg_val = 0
        best_iou_avg_test = 0
        best_tp_test = 0
        best_fp_test = 0
        best_fn_test = 0
        best_tp_val = 0
        best_fp_val = 0
        best_fn_val = 0
        best_mAP_val = 0
        best_mAP_test = 0
      
      separator = ";"
      print ("SUMMARY:")
      str_header = "Train" + separator
      str_header = "Val" + separator
      str_header = "Test" + separator
      str_header += "Window" + separator
      
      str_header += "Kernel" + separator
      str_header += "Features" + separator
      str_header += "Layers" + separator
      str_header += "Vertical reduction" + separator
      str_header += "Dropout" + separator
      
      
      str_header += "PAG" + separator
      str_header += "Num pages train" + separator
      str_header += "ANN" + separator
      str_header += "Num annotations per page" + separator
      str_header += "PAT" + separator
      str_header += "Num random patches" + separator
      str_header += "Th IoU" + separator
      str_header += "VAL" + separator
      str_header += "Th_bin" + separator
      str_header += "F1-val" + separator
      str_header += "P-val" + separator
      str_header += "R-val" + separator
      str_header += "TP-val" + separator
      str_header += "FP-val" + separator
      str_header += "FN-val" + separator
      str_header += "IoU-avg_val" + separator
      str_header += "Num annotated patches-val" + separator
      str_header += "Maximum num annotated patches-val" + separator
      str_header += "TEST" + separator
      str_header += "F1-test" + separator
      str_header += "P-test" + separator
      str_header += "R-test" + separator
      str_header += "TP-test" + separator
      str_header += "FP-test" + separator
      str_header += "FN-test" + separator
      str_header += "IoU-avg-test" + separator
      str_header += "Num annotated patches-test" + separator
      str_header += "Maximum num annotated patches-test" + separator
      str_header += "mAP-val" + separator
      str_header += "mAP-test" + separator

      str_properties = str(config.db_train_txt) + separator
      str_properties += str(config.db_val_txt) + separator
      str_properties += str(config.db_test_txt) + separator
      str_properties += str(input_shape[0]) + "x" + str(input_shape[1]) + separator
      str_properties += str(config.ker) + separator
      str_properties += str(config.nb_fil) + separator
      str_properties += str(config.n_la) + separator
      str_properties += str(config.vr) + separator
      str_properties += str(config.drop) + separator
      
      str_properties += "PAG" + separator
      str_properties += str(config.pages_train) + separator
      str_properties += "ANN" + separator
      str_properties += str(config.n_an) + separator
      str_properties += "PAT" + separator
      str_properties += str(config.n_pa) + separator
      str_result = str_properties
      str_result += str(config.iou) + separator
      str_result += "VAL"+separator
      str_result += str(best_th_val).replace(".",",") + separator
      str_result += number_to_string(best_fm_val) + separator
      str_result += number_to_string(prec_val) + separator
      str_result += number_to_string(recall_val) + separator  #number_to_string(best_fm_val) + separator + number_to_string(prec_val) + separator + number_to_string(recall_val) + separator + str(best_th_val).replace(".", ",") + separator
      str_result += str(best_tp_val).replace(".",",") + separator
      str_result += str(best_fp_val).replace(".",",") + separator
      str_result += str(best_fn_val).replace(".",",") + separator
      
      str_result += number_to_string(best_iou_avg_val) + separator
      str_result += str(number_annotated_patches_val) + separator
      str_result += str(max_number_annotated_patches_val) + separator

      print("Results: " + number_to_string(best_fm_test) + separator + number_to_string(prec_test) + separator + number_to_string(recall_test))
      
      str_result += "TEST" + separator
      str_result += number_to_string(best_fm_test) + separator 
      str_result += number_to_string(prec_test) + separator 
      str_result += number_to_string(recall_test) + separator
      str_result += str(best_tp_test).replace(".",",") + separator
      str_result += str(best_fp_test).replace(".",",") + separator
      str_result += str(best_fn_test).replace(".",",") + separator
      str_result += number_to_string(best_iou_avg_test) + separator
      
      str_result += str(number_annotated_patches) + separator
      str_result += str(max_number_annotated_patches) + separator
      str_result += number_to_string(best_mAP_val) + separator
      str_result += number_to_string(best_mAP_test) + separator
      
      
      if config.res is not None:
        utilIO.appendString(str_result, config.res, True)
      
        
      print ('*'*80)
      print(str_header)
      print(str_result)
      
    print("Finished")
    
    import signal
    pid = os.getpid()
    os.kill(pid, signal.SIGKILL) #or signal.SIGKILL 
