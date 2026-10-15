import json
import os
import numpy as np
import random
import matplotlib.pyplot as plt

# Configuration
import json
import os
import numpy as np
import random
import matplotlib.pyplot as plt

input_files = ["./Bongard-HOI/train/dataset-50.json", "./NLVR2/train/dataset-0.json", "./Co-Instruct-DB/train/dataset-0.json", "./TQA/train/dataset-0.json"]
input_files = ["./NLVR2/train/dataset-0.json", "./Bongard-HOI/train/dataset-50.json", "./HQ_Edit/train/dataset-0.json", "./PatternCom/train/dataset-0.json"]
input_files = ["./NLVR2/train/dataset-0.json", "./HQ_Edit/train/dataset-0.json", "./PatternCom/train/dataset-0.json", "./Bongard-HOI/train/dataset-2.json"]
input_files = ["./Bongard-HOI/train/dataset-50.json", "./Bongard-HOI/train/dataset-51.json", "./Bongard-HOI/train/dataset-52.json", "./Bongard-HOI/train/dataset-53.json"]
id = 50

output_prefix = "dataset"
sigma = 0.1  # Standard deviation for Gaussian noise
repeat = 5  # Number of phases for mixing
seed = 42  # Random seed for reproducibility

# Set seed for reproducibility
random.seed(seed)
np.random.seed(seed)

# Load datasets and get their sizes
datasets = []
dataset_sizes = []
for file in input_files:
    with open(file, "r") as f:
        data = json.load(f)
        datasets.append(data)
        dataset_sizes.append(len(data))

# Assign time distributions
num_datasets = len(datasets)
all_samples = []
colors = ["red", "blue", "green", "purple"]  # Colors for plotting

plt.figure(figsize=(10, 5))

for i, data in enumerate(datasets):
    num_samples = len(data)
    
    # Assign Gaussian-distributed time values over multiple repeats
    times = np.random.normal(i / num_datasets, sigma, size=num_samples)
    choices = np.random.choice(repeat, size=num_samples)  # Assign to different phases
    times += choices  # Adjust based on repeat phases
    
    # Shuffle data and assign timestamps
    random.shuffle(data)
    for j, sample in enumerate(data):
        if choices[j] <= i:  # Ensure sample appears within its allowed phase
            sample["time"] = times[j]
            all_samples.append(sample)
    
    # Plot histogram for visualization
    plt.hist(times, bins=30, alpha=0.5, label=f"Dataset {i+1}", color=colors[i])

# Shuffle and sort by time
random.shuffle(all_samples)
all_samples = sorted(all_samples, key=lambda d: d["time"])

# **Divide all_samples according to original dataset sizes**
start_idx = 0
for i, size in enumerate(dataset_sizes):
    subset = all_samples[start_idx:start_idx + size]
    start_idx += size
    
    # Save divided dataset
    output_file = f"./{input_files[i].split('/')[1]}/train/{output_prefix}-{id}.json"
    with open(output_file, "w") as f:
        json.dump(subset, f)
    
    print(f"Saved divided dataset {i+1} to {output_file}")

# Show distribution plot
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title(f"Gaussian Distributed Data Mixing (Repeat={repeat})")
plt.legend()
plt.savefig("a.jpg")



