

#%%
# Import Data of all participants into one big table

#import()



print("Hello World!")
#%%

import numpy as np
path_data = "/Users/antoniabielefeldt1/Documents/MaxPlanck/MultiEyeTem/Analysis/resultfile_pil05.txt"

# When you use names=True in numpy.genfromtxt(), the first row of the file is assumed to be the header row containing column names, 
# and these names are used to create fields in a structured numpy array. Each field corresponds to a column in the data, 
# and you can access the fields using their names.
single_behav_data = np.genfromtxt(path_data, delimiter='\t', names=True, dtype=None, encoding=None)

print(single_behav_data)

# make a list of all the files in the data folder

#%%
import os

file_list = os.listdir("/Users/antoniabielefeldt1/Documents/MaxPlanck/MultiEyeTem/Analysis")

print(file_list)


# %% Print out the variable names

variable_names = single_behav_data.dtype.names
print(variable_names)

#%% Print out specific data
print(behav_data['subID'])

#%%
# Calculate means for individual conditions



#%%
# Write a loop to import all behavioural data files and save them as individual variables?




# Concatenate all behavioural data files into one big table?