import numpy as np
from sklearn.metrics import roc_auc_score
import sys


def get_roc_auc_score(one_values, zero_values):
    roc_auc = roc_auc_score(np.concatenate([np.ones((one_values.shape[0])), np.zeros(
        (zero_values.shape[0]))]), np.concatenate([one_values, zero_values]))
    return roc_auc


result_path = sys.argv[1]
result = np.load(result_path)

true_labels = result["true_labels"]
original_probs = result["original_probs"]
adv_images = result["adv_images"]
adv_probs = result["adv_probs"]
repetition = 1 if not ("repetition" in result.keys()) else result["repetition"]
print(true_labels.shape)
print(original_probs.shape)
print(adv_probs.shape)
print(adv_images.shape)

FPR_labels = result["FPR_labels"]
FPR_probs = result["FPR_probs"]
print("len(FPR_labels)", len(FPR_labels))
NUM_CLASSES = len(np.unique(FPR_labels))

adv_probs = adv_probs[:, :NUM_CLASSES]
# Consider originally true for computing FPR% classification probability
fpr_originally_true_where = (FPR_labels == np.argmax(FPR_probs, axis=1))

FPR_labels = FPR_labels[fpr_originally_true_where]
FPR_probs = FPR_probs[fpr_originally_true_where]
print(len(np.unique(FPR_labels)))
# Compute FPR FPR% classification probability for each class
threshold_each_class = []
probs_each_class = []

print("NUM_CLASSES =", NUM_CLASSES, "\nrepetitions =", repetition)
if len(sys.argv) == 3:
    FPR = float(sys.argv[2])
else:
    FPR = 5

for c in range(NUM_CLASSES):
    probs = FPR_probs[FPR_labels == c, c]
    print(len(probs))

for c in range(NUM_CLASSES):
    probs = FPR_probs[FPR_labels == c, c]
    threshold_each_class.append(np.percentile(probs, FPR))
    probs_each_class.append(probs)

threshold_each_class = np.array(threshold_each_class)
# Consider originally true and undetected examples in computing robustness
originally_true_where = (true_labels == original_probs.argmax(axis=1))
originally_undetected_where = (original_probs[np.arange(original_probs.shape[0]), true_labels.astype(
    np.int64)] > threshold_each_class[true_labels.astype(np.int64)])
true_labels = true_labels[originally_true_where & originally_undetected_where]
original_probs = original_probs[originally_true_where &
                                originally_undetected_where]
original_preds = original_probs.argmax(axis=1)
# Apply repetition
originally_true_where = np.tile(
    originally_true_where[:, np.newaxis], (1, repetition)).reshape((-1, ))
originally_undetected_where = np.tile(
    originally_undetected_where[:, np.newaxis], (1, repetition)).reshape((-1, ))

adv_images = adv_images[originally_true_where & originally_undetected_where]
adv_probs = adv_probs[originally_true_where & originally_undetected_where]
adv_preds = adv_probs.argmax(axis=1)


adv_probs = adv_probs.reshape(-1, repetition, NUM_CLASSES)
adv_preds = adv_preds.reshape(-1, repetition)
adv_probs_new = []
adv_preds_new = []
for i in range(len(true_labels)):
    best_index = -1
    best_bypass_score = 9999999999
    tl = true_labels[i]
    for j in range(repetition):
        ap = adv_preds[i][j]
        # How small is this metric comapred to the threshold?
        # We negate threshold for unified approach for getting best score
        thrsh = -1 * np.abs(threshold_each_class[ap])
        bypass_score = adv_probs[i][j][ap] / thrsh

        if ap != tl and bypass_score < best_bypass_score:
            best_bypass_score = bypass_score
            best_index = j
    adv_probs_new.append(adv_probs[i][best_index])
    adv_preds_new.append(adv_preds[i][best_index])


adv_probs = np.array(adv_probs_new)
adv_preds = np.array(adv_preds_new)

# Compute 5 metrics including AUC score
incorrect_undetected = 0
incorrect_detected = 0
correct_undetected = 0
correct_detected = 0

expected_roc_score = 0

for c in range(NUM_CLASSES):
    threshold_c = threshold_each_class[c]

    adv_prob_c = adv_probs[adv_preds == c, c]
    bypass_criteria = adv_prob_c > threshold_c
    correct_criteria = true_labels[adv_preds == c] == c
    incorrect_undetected_whr_c = np.where(~correct_criteria & bypass_criteria)
    incorrect_detected_whr_c = np.where(~correct_criteria & ~bypass_criteria)
    correct_undetected_whr_c = np.where(correct_criteria & bypass_criteria)
    correct_detected_whr_c = np.where(correct_criteria & ~bypass_criteria)

    ones = np.ones(adv_probs[adv_preds == c].shape[0])
    incorrect_undetected += np.sum(ones[incorrect_undetected_whr_c])
    incorrect_detected += np.sum(ones[incorrect_detected_whr_c])
    correct_undetected += np.sum(ones[correct_undetected_whr_c])
    correct_detected += np.sum(ones[correct_detected_whr_c])

    weight = adv_prob_c.shape[0] / adv_probs.shape[0]

    if ones[~correct_criteria].sum() == 0:
        expected_roc_score += weight
    else:
        # Expected AUC ROC computation
        adv_prob_c_adjusted = np.copy(adv_prob_c)
        adv_prob_c_adjusted[correct_criteria] = np.min(adv_prob_c_adjusted)  # 0

        roc_auc = get_roc_auc_score(probs_each_class[c], adv_prob_c_adjusted)
        expected_roc_score += weight * roc_auc


total_length = adv_images.shape[0] / repetition
print("incorrect_undetected: {}".format(incorrect_undetected / total_length))
print("incorrect_detected: {}".format(incorrect_detected / total_length))
print("correct_undetected: {}".format(correct_undetected / total_length))
print("correct_detected: {}".format(correct_detected / total_length))

print("PRINT FOR COPY PASTE EPS: {}".format(result_path))
print("{}".format(incorrect_undetected / total_length))
print("{}".format(incorrect_detected / total_length))
print("{}".format(correct_undetected / total_length))
print("{}".format(correct_detected / total_length))
print("{}".format(expected_roc_score))
print("TFEC", threshold_each_class)
