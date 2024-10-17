from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score
import numpy as np
import torch
import matplotlib.pyplot as plt
 
def cal_multilabel_metrics(y_true, y_pre, labels, threshold=0.5):
    ''' Compute micro/macro AUROC and AUPRC
    
    :param y_true: Actual class labels
    :type y_true: torch.Tensor
    :param y_pre: Logits of predictions
    :type y_pre: torch.Tensor
    :param labels: Class labels used in the classification as SNOMED CT Codes
    :type labels: list
    
    :return: wanted metrics
    :rtypes: float
    '''

    # Convert tensors to numpy and filter out empty classes
    true_labels, pre_prob, _, _ = preprocess_labels(y_true, y_pre, labels, threshold)

    # ---------------- Wanted metrics ----------------

    # -- Average precision score
    macro_avg_prec = average_precision_score(true_labels, pre_prob, average = 'macro')
    micro_avg_prec = average_precision_score(true_labels, pre_prob, average = 'micro')
    
    # -- AUROC score
    micro_auroc = roc_auc_score(true_labels, pre_prob, average = 'micro')
    macro_auroc = roc_auc_score(true_labels, pre_prob, average = 'macro')

    return macro_avg_prec, micro_avg_prec, macro_auroc, micro_auroc

    
def preprocess_labels(y_true, y_pre, labels, threshold = 0.5, drop_missing = True):
    ''' Convert tensor variables to numpy and check the positive class labels. 
    If there's none, leave the columns out from actual labels, binary predictions,
    logits and class labels used in the classification.
    
    :param y_true: Actual class labels
    :type y_true: torch.Tensor
    :param y_pre: Logits of predicted labels
    :type y_pre: torch.Tensor
    
    :return true_labels, pre_prob, pre_binary, labels: Converted (and possibly filtered) actual labels,
                                                       binary predictions and logits

    :rtype: numpy.ndarrays
    '''

    # Actual labels from tensor to numpy
    true_labels = y_true.cpu().detach().numpy().astype(np.int32)  

    # Logits from tensor to numpy
    pre_prob = y_pre.cpu().detach().numpy().astype(np.float32)
    
    # ------ One-hot-endcode predicted labels ------

    pre_binary = np.zeros(pre_prob.shape, dtype=np.int32)

    # Find the index of the maximum value within the logits
    likeliest_dx = np.argmax(pre_prob, axis=1)

    # First, add the most likeliest diagnosis to the predicted label
    pre_binary[np.arange(true_labels.shape[0]), likeliest_dx] = 1

    # Then, add all the others that are above the decision threshold
    other_dx = pre_prob >= threshold

    pre_binary = pre_binary + other_dx
    pre_binary[pre_binary > 1.1] = 1
    pre_binary = np.squeeze(pre_binary) 

    if drop_missing:
        
         # ------ Check the positive class labels ------
    
        # Find all the columnwise indexes where there's no positive class
        null_idx = np.argwhere(np.all(true_labels[..., :] == 0, axis=0))

        # Drop the all-zero columns from actual labels, logits,
        # binary predictions and class labels used in the classification
        if any(null_idx):
            true_labels = np.delete(true_labels, null_idx, axis=1)
            pre_prob = np.delete(pre_prob, null_idx, axis=1)
            pre_binary = np.delete(pre_binary, null_idx, axis=1)
            labels = np.delete(labels, null_idx)

    # There should be as many actual labels and logits as there are labels left
    assert true_labels.shape[1] == pre_prob.shape[1] == pre_binary.shape[1] == len(labels)
    
    return true_labels, pre_prob, pre_binary, labels


def roc_curves(y_true, y_pre, labels, epoch=None, save_path='./experiments/'):
    '''Compute and plot the ROC Curves for each class, also macro and micro. Save as a png image.
    
    :param y_true: Actual labels
    :type y_true: torch.Tensor
    :param y_pred: Logits of predicted labels
    :type y_pred: torch.Tensor
    :param labels: Class labels used in the classification as SNOMED CT Codes
    :type labels: list
    :param epoch: Epoch in which the predictions are made
    :type epoch: int
    '''

    # Convert tensors to numpy and filter out empty classes
    true_labels, pre_prob, _, cls_labels = preprocess_labels(y_true, y_pre, labels, drop_missing=True)
    
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # AUROC, fpr and tpr for each label
    for i in range(len(cls_labels)):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], pre_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), pre_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Interpolate all ROC curves at these points to compute macro-average ROC area
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(len(cls_labels)):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average the mean TPR and compute AUC
    mean_tpr /= len(cls_labels)
    
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))
    fig.suptitle('ROC Curves')

    # Plotting micro-average and macro-average ROC curves
    ax1.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

    ax1.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]))

    # Plotting ROCs for each class
    for i in range(len(cls_labels)):
        ax2.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(cls_labels[i], roc_auc[i]))

    # Adding labels and titles for plots
    ax1.plot([0, 1], [0, 1], 'k--'); ax2.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0]); ax2.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05]); ax2.set_ylim([0.0, 1.05])
    ax1.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax2.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax1.legend(loc="lower right", prop={'size': 8}); ax2.legend(loc="lower right", prop={'size': 6})
    
    fig.tight_layout()
    # Saving the plot 
    name = "roc-e{}.png".format(epoch) if epoch else "roc-test.png"
    
    plt.savefig(save_path + '/' + name, bbox_inches = "tight")
    plt.close(fig) 


if __name__ == '__main__':

    y_actual = torch.Tensor([[0, 0, 1, 1], [0, 1, 0, 0], [1, 0, 1, 0]])
    y_prob = torch.Tensor([[0.9, 0.8, 0.56, 0.8], [0.9, 0.8, 0.8, 0.6], [0.9, 0.7, 0.56, 0.8]])
    labels = ['164889003', '164890007', '6374002', '733534002']

    cal_multilabel_metrics(y_actual, y_prob, labels, threshold=0.5)