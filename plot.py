import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in ((i,j) for i in range(cm.shape[0]) for j in range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Define the confusion matrix

confusion_matrix = np.array(x)
#plt.figure(figsize=(10, 8))
#plot_confusion_matrix(confusion_matrix, classes=[str(i) for i in range(10)],
                      #title='Confusion matrix, without normalization')

#plt.show()

import numpy as np

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    total_sum = confusion_matrix.sum()
    return diagonal_sum / total_sum

def precision(confusion_matrix, class_label):
    col = confusion_matrix[:, class_label]
    return confusion_matrix[class_label, class_label] / col.sum()

def recall(confusion_matrix, class_label):
    row = confusion_matrix[class_label, :]
    return confusion_matrix[class_label, class_label] / row.sum()

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

# Calculate accuracy
acc = accuracy(confusion_matrix)
print("Accuracy:", acc)

# Calculate macro F1 score
f1_scores = []
for i in range(len(confusion_matrix)):
    precision_i = precision(confusion_matrix, i)
    recall_i = recall(confusion_matrix, i)
    f1_i = f1_score(precision_i, recall_i)
    f1_scores.append(f1_i)

macro_f1 = np.mean(f1_scores)
print("Macro F1 score:", macro_f1)

