

#%%
# Import Data of all participants into one big table

#import()



print("Hello World!")
#%%

import numpy as np
# general path to behavioural data folder
path_data = "/Users/antoniabielefeldt1/Documents/MaxPlanck/MultiEyeTem/Analysis/MultiEyeTem_Repo/behavdata"

# When you use names=True in numpy.genfromtxt(), the first row of the file is assumed to be the header row containing column names, 
# and these names are used to create fields in a structured numpy array. Each field corresponds to a column in the data, 
# and you can access the fields using their names.

# need to change the following line because path data is now a general path to the data folder. Is there a possibility to add the specific file I want
# to import for a single analysis to the path? Something like path_data + resultfilepil05
single_behav_data = np.genfromtxt(path_data, delimiter='\t', names=True, dtype=None, encoding=None)

print(single_behav_data)

# make a list of all the files in the data folder

#%%
import os

file_list = os.listdir("/Users/antoniabielefeldt1/Documents/MaxPlanck/MultiEyeTem/Analysis")

print(file_list)

# git check
hgit

# %% Print out the variable names

variable_names = single_behav_data.dtype.names
print(variable_names)

#%% Print out specific data
print(behav_data['subID'])

#%%
# Calculate means for individual conditions



#%%
# Write a loop to import all behavioural data files and save them as individual variables?


#not used yet/copied from ChatGPT
import numpy as np
import os

# Set the path to the directory containing the text files
path = "/path/to/directory"

# Get a list of all the .txt files in the directory
file_list = [f for f in os.listdir(path) if f.endswith('.txt')]

# Loop over each file and import it as a numpy array using numpy.genfromtxt()
for file_name in file_list:
    file_path = os.path.join(path, file_name)  # Get the full path of the file
    data = np.genfromtxt(file_path, delimiter=',', names=True, dtype=None, encoding=None)
    # Do something with the imported data (e.g., print it)
    print(f"Data from {file_name}:")
    print(data)


# Concatenate all behavioural data files into one big table?







#%% Behavioural Analyses



# Means of different conditions


# Are participants better for upright positions than upside down positions of objects?


# Cardinal Bias??





#%% Plot data
#Plot Means
