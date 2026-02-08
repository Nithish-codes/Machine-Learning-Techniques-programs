import csv

def load_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        header = data[0]
        examples = data[1:]
    return header, examples

def candidate_elimination(examples):
    num_attributes = len(examples[0]) - 1
    
    # Initialize S with the first positive example
    S = ['0'] * num_attributes
    for ex in examples:
        if ex[-1].lower() == "yes":
            S = ex[:-1]
            break
    
    # Initialize G with the most general hypothesis
    G = [['?' for _ in range(num_attributes)]]

    for i, example in enumerate(examples):
        attributes, label = example[:-1], example[-1].lower()

        if label.lower() == "yes":  # Positive example
            # 1. Generalize S to include the positive example
            for j in range(num_attributes):
                if S[j] != attributes[j]:
                    S[j] = '?'
            
            # 2. Remove hypotheses from G that are inconsistent with the positive example
            # (i.e., remove g if it does NOT cover this positive example)
            G = [g for g in G if all(g[k] == '?' or g[k] == attributes[k] for k in range(num_attributes))]

        else:  # Negative example
            new_G = []
            for g in G:
                # Check if hypothesis g covers the negative example (which is bad)
                covers_negative = all(g[k] == '?' or g[k] == attributes[k] for k in range(num_attributes))

                if covers_negative:
                    # g is too general (covers negative), so we must specialize it
                    for k in range(num_attributes):
                        if g[k] == '?':
                            if S[k] != '?' and S[k] != attributes[k]:
                                new_hypothesis = g.copy()
                                new_hypothesis[k] = S[k]
                                
                                # Verify the new hypothesis covers S before adding
                                if all(new_hypothesis[j] == '?' or new_hypothesis[j] == S[j] for j in range(num_attributes)):
                                    if new_hypothesis not in new_G:
                                        new_G.append(new_hypothesis)
                else:
                    # g does NOT cover the negative example, so it is safe to keep
                    new_G.append(g)
            
            G = new_G

    return S, G

if __name__ == "__main__":
    try:
        header, examples = load_data("training_data_set_for_ex_1.csv")
        S, G = candidate_elimination(examples)
        
        print("\n--- Final Result ---")
        print("Final Specific Boundary (S):", S)
        print("Final General Boundary (G):", G)
    except FileNotFoundError:
        print("Error: 'training_data.csv' not found. Please create the file first.")
