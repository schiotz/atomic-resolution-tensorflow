from stm.preprocess import normalize
from stm.feature.peaks import find_local_peaks, refine_peaks
from skimage.morphology import disk
from scipy.spatial import cKDTree as KDTree

def precision_recall(predicted, target, sampling):
    """Precision and recall for peak positions"""
    # Precision: Number of correctly predicted peaks 
    # divided by number of target peaks
    distance=0.5/sampling
    if len(predicted) == 0:
        return (0.0, 1.0)
    if len(target) == 0:
        return (1.0, 0.0)
    tree = KDTree(target)
    x = tree.query(predicted, distance_upper_bound=distance)[0]
    precision = (x <= distance).sum() / len(predicted)
    # Recall: Number of target peaks that were found
    # divided by total number of target peaks
    tree = KDTree(predicted)
    x = tree.query(target, distance_upper_bound=distance)[0]
    recall = (x <= distance).sum() / len(target)
    return (precision, recall)

def evaluate_result(inference, label, sampling, accept_distance=2.0, threshold=0.6):
    "Evaluate the prediction for an image."
    distance = int(accept_distance / sampling)
    # Find the peaks
    infer_peaks = find_local_peaks(inference[:,:,0], min_distance=distance, 
                                   threshold=threshold, exclude_border=10,
                                   exclude_adjacent=True)
    label_peaks = find_local_peaks(label[:,:,0], min_distance=distance, 
                                   threshold=threshold, exclude_border=10,
                                   exclude_adjacent=True)

    # Refine the peaks
    region = disk(2)
    infer_refined = refine_peaks(normalize(inference[:,:,0]), infer_peaks, 
                                region, model='polynomial')
    label_refined = refine_peaks(normalize(label[:,:,0]), label_peaks, 
                                region, model='polynomial')
    return precision_recall(infer_refined, label_refined, sampling)
