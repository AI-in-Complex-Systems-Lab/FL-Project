import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

folder_path = 'data/cifar-10-batches-py'
file_name = 'data_batch_1'
file_path = os.path.join(folder_path, file_name)

with open(file_path, 'rb') as file:
    data_dict = pickle.load(file, encoding='bytes')

images = data_dict[b'data']
labels = data_dict[b'labels']


# print(f'Number of images: {len(images)}')
# print(f'Number of labels: {len(labels)}')

first_image = images[0]  # A single image
first_label = labels[0]  # Corresponding label  

# print(f'First image shape: {first_image.shape}')
# print(f'First image label: {first_label}')

first_image = first_image.reshape(3, 32, 32).transpose(1, 2, 0)  # Reshape to 32x32x3
# print(f'First image pixel values:\n{first_image}')












# # show_data.py
# import os
# import pickle

# # # Construct the file path
# folder_path = 'data/cifar-10-batches-py'
# file_name = 'data_batch_1'
# file_path = os.path.join(folder_path, file_name)

# # # Read the file
# with open(file_path, 'rb') as file:
#     data = file.read()
# # # Now `data` contains the contents of the file.
# # You can process it as needed here.

import os
import pickle
import numpy as np        

# CIFAR-10 class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Construct the file path
folder_path = 'data/cifar-10-batches-py'
file_name = 'data_batch_1'
file_path = os.path.join(folder_path, file_name)

# Read and unpickle the file
with open(file_path, 'rb') as file:
    batch = pickle.load(file, encoding='bytes')

# Extract the data from the batch
data = batch[b'data']
labels = batch[b'labels']

# Create a dictionary to store image info
images_info = {}

# Populate the dictionary with image index, label, class name, and a small portion of data (first few pixels)
for i in range(len(data)):
    # Convert the i-th image's data to a numpy array and reshape it to 32x32x3
    image_data = list(np.array(data[i]).reshape(3, 32, 32).transpose(1, 2, 0).flatten()[:10])  # First 10 pixels for brevity
    
    # Get the class name from the label
    class_name = class_names[labels[i]]
    
    images_info[f'Image_{i+1}'] = {
        'Label': labels[i],
        'Class Name': class_name,
        'Data (first 10 pixels)': image_data
    }

#Print the dictionary
# for image, info in list(images_info.items())[:5]:
#     print(f"{image}:")
#     print(f"  Label: {info['Label']}")
#     print(f"  Class Name: {info['Class Name']}")
#     print(f"  Data (first 10 pixels): {info['Data (first 10 pixels)']}")
#     print()


import random



# Number of images to poison
num_poisoned = 1000



# Create a copy of labels to modify
poisoned_labels = labels.copy()



# Randomly choose indices to poison
indices = random.sample(range(len(labels)), num_poisoned)




for idx in indices:
    # Flip the label with another random class
    current_label = labels[idx]
    new_label = (current_label + random.randint(1, 9)) % 10  # Ensures a different class
    poisoned_labels[idx] = new_label




# Update the dictionary with poisoned data
poisoned_images_info = {}
for i in range(len(data)):
    image_data = list(np.array(data[i]).reshape(3, 32, 32).transpose(1, 2, 0).flatten()[:10])  # First 10 pixels for brevity
    class_name = class_names[poisoned_labels[i]]
    poisoned_images_info[f'Image_{i+1}'] = {
        'Label': poisoned_labels[i],
        'Class Name': class_name,
        'Data (first 10 pixels)': image_data
    }

#Print a few entries to see the effect
# for image, info in list(poisoned_images_info.items())[:5]:
#     print(f"{image}:")
#     print(f"  Label: {info['Label']}")
#     print(f"  Class Name: {info['Class Name']}")
#     print(f"  Data (first 10 pixels): {info['Data (first 10 pixels)']}")
#     print()



# # # show_data.py
# import os
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt

# # Construct the file path
# folder_path = 'data/cifar-10-batches-py'
# file_name = 'data_batch_1'
# file_path = os.path.join(folder_path, file_name)

# # Read the binary file using pickle
# with open(file_path, 'rb') as file:
#     data_dict = pickle.load(file, encoding='bytes')

# # Extract images and labels
# images = data_dict[b'data']
# labels = data_dict[b'labels']

# # Convert images to a NumPy array
# images = np.array(images)

# # Print the shape of the images to verify
# # print(f'Images shape: {images.shape}')  # Should be (10000, 3072)

# # Display the first image
# first_image = images[0]  # Get the first image
# first_label = labels[321]  # Corresponding label

# # Reshape the image to 32x32x3
# first_image = first_image.reshape(3, 32, 32).transpose(1, 2, 0)  # Convert to 32x32x3 format

# # Normalize pixel values to the range 0-1 for display
# first_image = first_image / 255.0

# #Display the image
# plt.imshow(first_image)
# plt.title(f'Label: {first_label}')
# plt.axis('off')  # Hide axis ticks and labels
# plt.show()






