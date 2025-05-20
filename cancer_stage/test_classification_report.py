import numpy as np
from sklearn.metrics import classification_report

# Test data
all_labels = np.array([0, 1, 2, 0, 1, 2])
all_preds = np.array([0, 1, 2, 1, 0, 2])
class_names = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']

# Get unique classes present in the data
unique_classes = sorted(np.unique(all_labels))
# Create class names dynamically based on actual classes present
present_class_names = [class_names[i] for i in unique_classes]

print('Unique classes:', unique_classes)
print('Present class names:', present_class_names)
print(classification_report(all_labels, all_preds, target_names=present_class_names))
