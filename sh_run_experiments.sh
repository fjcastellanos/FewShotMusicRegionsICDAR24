#!/bin/bash

MODE="train" # "test" or "train"
GPU=0  # index of the GPU to be used
TYPE="CNN"


AUG="random rot scale"  # 'random', 'flipH', 'flipV', 'rot', 'scale'


db_train_txt="datasets/Folds/b-59-850/fold0/train.txt"  #List of JSON data (from MURET) to be used for training
db_val_txt="datasets/Folds/b-59-850/fold0/val.txt"      #List of JSON data (from MURET) to be used for validation
db_test_txt="datasets/Folds/b-59-850/fold0/test.txt"    #List of JSON data (from MURET) to be used for testing. 
CLASSES="staff empty_staff"  #Classes to be considered.
IOU_THRESHOLD=0.5 # Threshold to consider a prediction as True Positive


EPOCHS=200 # Maximum number of epochs.
BATCH_SIZE=32 # Size of the batch
PATHRESULTS="results/res_experiment.txt" # Output file to save results
OPTIONS="-adapt-size-patch"    #"--test",  "-adapt-size-patch"

aug_serial=${AUG// /.}
aug_serial=${aug_serial////-}

options_serial=${OPTIONS// /.}
options_serial=${options_serial////-}

RUN=1
COUNTER=0

mkdir logs/${MODE}/
for FILTERS in 32; do
    for WINDOW_W in 256; do
        for WINDOW_H in 256; do
            for LAYERS in 3; do
                for KERNEL_SIZE in 3; do
                    for DROPOUT in 0.2; do
                        for PAGES_TRAIN in -1; do
                            for NUMBER_ANNOTATED_PATCHES in -1 1 2 4 8 16 32; do
                                for NUMBER_PATCHES in 2048; do #
                                    for source in "b-59-850" "SEILS" "Guatemala" "Mus-Tradicional/c-combined" "Mus-Tradicional/m-combined" "Patriarca" ; do #
                                        SOURCE_PATH_TRAIN="datasets/Folds/"${source}"/fold0/train.txt"
                                        SOURCE_PATH_VAL="datasets/Folds/"${source}"/fold0/val.txt"
                                        SOURCE_PATH_TEST="datasets/Folds/"${source}"/fold0/test.txt"

                                        for VERTICAL_REDUCTION in 0.4; do #
                                            target=${source}

                                            output_file="out_${TYPE}_vr_${VERTICAL_REDUCTION}_${source}_aug${aug_serial}_w${WINDOW_W}_h${WINDOW_H}_l${LAYERS}_f${FILTERS}_k${KERNEL_SIZE}_d${DROPOUT}_pt${PAGES_TRAIN}_np${NUMBER_PATCHES}_nap${NUMBER_ANNOTATED_PATCHES}_e${EPOCHS}_b${BATCH_SIZE}_${options_serial}.txt"

                                            output_file=${output_file// /.}
                                            output_file=${output_file////-}
											output_file="logs/${MODE}/${output_file}"                              
											echo $output_file

                                            python -u main.py \
                                                        -db_train_txt ${SOURCE_PATH_TRAIN} \
                                                        -db_val_txt ${SOURCE_PATH_VAL} \
                                                        -db_test_txt ${SOURCE_PATH_TEST} \
                                                        -cls ${CLASSES} \
                                                        -aug ${AUG} \
                                                        -window_w ${WINDOW_W} \
                                                        -window_h ${WINDOW_H} \
                                                        -l ${LAYERS} \
                                                        -f ${FILTERS} \
                                                        -k ${KERNEL_SIZE} \
                                                        -drop ${DROPOUT} \
                                                        -npatches ${NUMBER_PATCHES} \
                                                        -n_annotated_patches ${NUMBER_ANNOTATED_PATCHES} \
                                                        -pages_train ${PAGES_TRAIN} \
                                                        -e ${EPOCHS} \
                                                        -b ${BATCH_SIZE} \
                                                        -gpu ${GPU} \
                                                        -res ${PATHRESULTS} \
                                                        -iou ${IOU_THRESHOLD} \
                                                        -vr ${VERTICAL_REDUCTION} \
                                                        ${OPTIONS} \
                                                        &> ${output_file}

											echo "Finished\n"
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done        
        done
    done
done


