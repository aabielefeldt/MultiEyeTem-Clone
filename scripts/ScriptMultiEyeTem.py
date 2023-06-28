
#%%

# at the moment, this is just to try out how to use the directory and show my files
# this however at the moment shows only the directory to the script file I am writing at the moment
# I want to transform this, so I have access to the data files within the MultiEyeTem_Repo folder
# Ideally there is probably a way to use the project package I created to do that, or maybe even that is the point of the project package and I do not even have to do anything
# maybe I need to make clear at the beginning of the code that I am using that package (something like add_path or something)
import os
current_directory = os.getcwd()
print("Current Directory:", current_directory)

files = os.listdir()
print(files)



#%%
# Import Data of all participants into one big table
# do I even need to do that when all my data is in the folder that I am working in anyway? (i.e. my repo?)

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

file_list = os.listdir("/Users/antoniabielefeldt1/Documents/MaxPlanck/MultiEyeTem/Analysis/MultiEyeTem_Repo")

print(file_list)

# git check, edit: what was I trying to do here??
#hgit

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
import pandas as pd

import os
current_directory = os.getcwd()
print("Current Directory:", current_directory)

files = os.listdir()
print(files)

data = pd.read_csv(current_directory)
#%%
for file_name in data #no idea if these are already used and therefore work but should be somthing looking like this
    acc_cond1 = data['acc_trial_concrete_cued'].mean()
    #acc_cond2 = mean()
    #acc_cond3 = mean()

# Are participants better for upright positions than upside down positions of objects?


# Cardinal Bias??

#%%

numbers = [1, 2, 3, 4, 5]

for whatever in numbers:
    print(whatever)


#%% Plot data
#Plot Means

# table of accuracies for 1. every condition, 2. every position, 3. upside down vs. right side up, 4. mnemonic distance, 5. condition plus mnemonic distance

# same table but with reaction times instead of accuracy 
# *hello*
## heading
