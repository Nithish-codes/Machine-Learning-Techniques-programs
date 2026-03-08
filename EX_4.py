import csv
import math

def load_csv(filename):
    with open(filename, 'r') as file:
        lines = csv.reader(file)
        dataset = list(lines)
    header = dataset.pop(0)
    return dataset, header

def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    return dataset[:train_size], dataset[train_size:]

def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        class_val = vector[-1]
        if class_val not in separated:
            separated[class_val] = []
        separated[class_val].append(vector)
    return separated

def calculate_attribute_probabilities(dataset):
    """Calculates the frequency of each attribute value per class."""
    probabilities = {}
    total_rows = len(dataset)
    for i in range(len(dataset[0]) - 1): # Exclude target column
        probabilities[i] = {}
        for row in dataset:
            val = row[i]
            probabilities[i][val] = probabilities[i].get(val, 0) + 1
        
        # Convert counts to probabilities
        for val in probabilities[i]:
            probabilities[i][val] /= total_rows
    return probabilities, total_rows

def predict(summaries, class_probs, input_vector):
    """Calculates the probability of each class for a given input."""
    probabilities = {}
    for class_val, data in summaries.items():
        # Start with the prior probability: P(Class)
        probabilities[class_val] = class_probs[class_val]
        
        # Multiply by conditional probabilities: P(Feature | Class)
        attr_probs = data['attr_probs']
        for i in range(len(input_vector)):
            val = input_vector[i]
            # Use a very small probability if value was never seen (Laplace smoothing lite)
            probabilities[class_val] *= attr_probs[i].get(val, 0.0001)
            
    return max(probabilities, key=probabilities.get)

def get_accuracy(test_set, summaries, class_probs):
    correct = 0
    for row in test_set:
        prediction = predict(summaries, class_probs, row[:-1])
        if prediction == row[-1]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0

# --- Execution ---
dataset, header = load_csv('training_data_set_for_ex_4.csv')
train_set, test_set = split_dataset(dataset, 0.8)

# Group training data by class
separated = separate_by_class(train_set)
summaries = {}
class_probs = {}
total_train = len(train_set)

for class_val, instances in separated.items():
    attr_probs, count = calculate_attribute_probabilities(instances)
    summaries[class_val] = {'attr_probs': attr_probs}
    class_probs[class_val] = count / total_train

# Evaluate
accuracy = get_accuracy(test_set, summaries, class_probs)

print(f"Dataset loaded with {len(dataset)} rows.")
print(f"Training on {len(train_set)} rows, Testing on {len(test_set)} rows.")
print(f"Accuracy: {accuracy}%")

# Test with a new sample
new_sample = ['High', 'Low', 'Yes']
result = predict(summaries, class_probs, new_sample)
print(f"New Sample {new_sample} predicted as: {result}")
