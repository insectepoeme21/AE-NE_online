
import os
import cv2
import natsort
import numpy as np

def compute_confusion_matrix(dataset_name,foreground_mask, ground_truth_input, use_roi = False, roi_mask = None):

    if dataset_name == 'LASIESTA':
        # cf definition of ground_truth_inputs on LASIESTA website
        # conversion to format 255=foreground, 0=background 128 = undefined
        max_ground_truth_input = np.amax(ground_truth_input, axis=2)
        undefined_class_mask = (np.sum(ground_truth_input, axis=2) == 765).astype(np.int)
        ground_truth_input = undefined_class_mask * 128 + (1 - undefined_class_mask) * max_ground_truth_input
    elif dataset_name == "CDnet":
        if ground_truth_input.ndim == 3:
            ground_truth_input = np.amax(ground_truth_input, axis=2)
        if roi_mask.shape != ground_truth_input.shape:# roi shape for "traffic" video has wrong shape
            use_roi = False

    true_mask = (foreground_mask == 255)
    false_mask = (foreground_mask == 0)
    true_GT = (ground_truth_input == 255)
    false_GT = (ground_truth_input == 0)

    if use_roi:
        true_mask = np.logical_and(true_mask, roi_mask)
        false_mask = np.logical_and(false_mask, roi_mask)
        true_GT = np.logical_and(true_GT, roi_mask)
        false_GT = np.logical_and(false_GT, roi_mask)

    tp = np.sum(np.logical_and(true_mask,true_GT).astype(int))
    fp = np.sum(np.logical_and(true_mask,false_GT).astype(int))
    tn = np.sum(np.logical_and(false_mask, false_GT).astype(int))
    fn = np.sum(np.logical_and(false_mask, true_GT).astype(int))

    return tp, tn, fp, fn

def compute_statistics(dataset_name, video_name, masks_path, GTs_path, roi_path=None, temporal_roi_path=None):

            mask_ids = natsort.natsorted([i for i in os.listdir(masks_path) if os.path.splitext(i)[-1] == ".png"])
            GT_ids = natsort.natsorted([i for i in os.listdir(GTs_path) if os.path.splitext(i)[-1] == ".png"])
            
            first_mask_idx = int(os.path.splitext(mask_ids[0])[0].replace("bin", "")) - 1
            last_mask_idx = int(os.path.splitext(mask_ids[-1])[0].replace("bin", ""))
            first_GT_idx = int(os.path.splitext(GT_ids[0])[0].replace("gt", "")) - 1
            last_GT_idx = int(os.path.splitext(GT_ids[-1])[0].replace("gt", ""))

            if first_mask_idx > first_GT_idx or last_GT_idx > last_mask_idx:
                 GT_ids = GT_ids[first_mask_idx:last_mask_idx]
            #print(mask_ids[0], mask_ids[-1], GT_ids[0], GT_ids[-1])

            roi_mask = None
            use_roi = False

            if temporal_roi_path != None: # for CDnet , which uses temporal roi

                assert os.path.isfile(temporal_roi_path), f"error, no temporal roi at {temporal_roi_path}"
                f = open(temporal_roi_path, "r")
                start_idx, end_idx = f.readline().split()
                start_idx = int(start_idx)
                end_idx = int(end_idx)
                f.close()
                if start_idx > first_mask_idx:
                    start = start_idx - first_mask_idx
                    end = end_idx - first_mask_idx + 1
                    mask_ids = mask_ids[start:end]
                    GT_ids = GT_ids[start:end]

            if roi_path != None:  # for CDnet , which uses spatial roi
                assert os.path.isfile(roi_path), f"error, no roi found at {roi_path}"
                print('using roi')
                roi = cv2.imread(roi_path, cv2.IMREAD_UNCHANGED)
                roi = np.asarray(roi)
                roi_mask = (roi == 255)
                use_roi = True

            TP = 0
            TN = 0
            FP = 0
            FN = 0
            lst = []
            for mask_id, GT_id in zip(mask_ids, GT_ids):
                mask_path = os.path.join(masks_path, mask_id)
                GT_path = os.path.join(GTs_path, GT_id)
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                ground_truth_input = cv2.imread(GT_path, cv2.IMREAD_UNCHANGED)  # HWC

                tp, tn, fp, fn = compute_confusion_matrix(dataset_name,mask, ground_truth_input, use_roi,roi_mask)
                TP += tp
                TN += tn
                FP += fp
                FN += fn
                fm = tp / (tp + 0.5 * (fp + fn))
                lst.append(fm)
                #print(mask_id, GT_id, tp, tn, fp, fn, fm)

            if TP + FP + FN > 0:
                FM = TP / (TP + 0.5 * (FP + FN))
            else:
                print('warning : no foreground object in sequence, 100% true negatives !')
                FM = 1
            recall = TP/(TP+FN)
            precision = TP/(TP+FP)
            statistics = f'video {video_name} :  FM={FM}, precision = {precision} recall = {recall} TP = {TP} TN = {TN} FP= {FP},FN= {FN} '
            return statistics, lst
