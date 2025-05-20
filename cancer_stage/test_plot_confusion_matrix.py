import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    # Get unique classes present in the data
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    # Create class names dynamically based on actual classes present
    present_class_names = [class_names[i] for i in unique_classes]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(present_class_names))
    plt.xticks(tick_marks, present_class_names, rotation=45)
    plt.yticks(tick_marks, present_class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_test.png')
    plt.show()

# Test data
all_labels = np.array([0, 1, 2, 0, 1, 2])
all_preds = np.array([0, 1, 2, 1, 0, 2])
class_names = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']

# Plot confusion matrix
plot_confusion_matrix(all_labels, all_preds, class_names)

print('Unique classes:', sorted(np.unique(np.concatenate([all_labels, all_preds]))))
print('Present class names:', [class_names[i] for i in sorted(np.unique(np.concatenate([all_labels, all_preds])))])
