#!/bin/bash

MODE="train"
GPU=0
TYPE="CNN"


AUG="random rot scale"  # 'all', 'none', 'flipH', 'flipV', 'wb', 'expos', 'rot', 'scale', 'blur', 'dropout'

WINDOW_W=256
WINDOW_H=256

LAYERS=4
FILTERS=64
KERNEL_SIZE=5
DROPOUT=0.2

PAGES_TRAIN=-1
NUMBER_PATCHES=2048
NUMBER_ANNOTATED_PATCHES=1


db_train_txt="datasets/Folds/b-59-850/fold0/train.txt"
db_val_txt="datasets/Folds/b-59-850/fold0/val.txt"
db_test_txt="datasets/Folds/b-59-850/fold0/test.txt" 
CLASSES="staff empty_staff"
IOU_THRESHOLD=0.5

window_w="256" 
window_h="256" 

VERTICAL_REDUCTION="0.4"

EPOCHS=200
BATCH_SIZE=32
VERBOSE=1
PATHRESULTS="results/res_final_experiment_with_adapt_size.txt"
OPTIONS="-adapt-size-patch"    #--test

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

                                            let COUNTER++

                                            echo ${COUNTER}"\n"${RUN}

                                            if [[ $NUMBER_PATCHES -eq 512 && $PAGES_TRAIN -eq 1 && $NUMBER_ANNOTATED_PATCHES -eq -1 &&  "$source" = "b-59-850" && $VERTICAL_REDUCTION -eq 0.0 ]] ; then
                                                let RUN=1
                                            fi
                                            if [ ${RUN} -eq 1 ] ; then
                                                echo 'RUN!!!\n'
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
                                            fi
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


