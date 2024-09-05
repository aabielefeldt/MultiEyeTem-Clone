# %% 
# Imports
import pandas as pd

# %% Loading Data

df = pd.read_csv('../data/csv_files/df.csv')
good_partis = df['subID'].unique()
# %% 
# Descriptive Statistics after Exclusion
descriptive_stats_after_exclusion = {}

# Number of participants after exclusion
len_good_partis = len(good_partis)
descriptive_stats_after_exclusion['Total Participants'] = len_good_partis
print(f'There were {len_good_partis} participants in the dataset that went into the final analysis.')

# Gender
# Group the DataFrame by subID and get the first occurrence of sub_gender in each group
gender_by_sub_id = df.groupby('subID')['sub_gender'].first()

# Count the instances of 'M' within the grouped sub_gender data
gender_counts = gender_by_sub_id.value_counts()
if 'M' in gender_counts:
    count_of_males = gender_counts['M']
    descriptive_stats_after_exclusion['Male Participants'] = count_of_males
    print(f"Count of male Participants: {count_of_males}")

# Calculate the proportion of males
gender_propo = count_of_males / len(good_partis)
descriptive_stats_after_exclusion['Proportion of Male Participants'] = gender_propo
print(f'Proportion of male participants: {gender_propo:.2f}')

# Age
age_by_sub_id = df.groupby('subID')['sub_age'].first()
mean_age = age_by_sub_id.mean()
sd_age = age_by_sub_id.std()
descriptive_stats_after_exclusion['Mean Age'] = mean_age
descriptive_stats_after_exclusion['Standard Deviation Age'] = sd_age
print(f'The mean age of participants was {mean_age:.2f} with the standard deviation of {sd_age:.2f}')

# Number of participants with and without mask
masked_pp = df.groupby('subID')['mask'].first()
mask_counts = masked_pp.value_counts()
descriptive_stats_after_exclusion['Masked Participants'] = mask_counts[1]
descriptive_stats_after_exclusion['Unmasked Participants'] = mask_counts[0]
print(f"There were {mask_counts[1]} participants with the mask and {mask_counts[0]} without the mask.")

# Number of participants per stimuli set
stimset_pp = df.groupby('subID')['stimuli_set'].first()
stimset_counts = stimset_pp.value_counts()
descriptive_stats_after_exclusion['Stimuli Set 1 Participants'] = stimset_counts['stimuli_set1']
descriptive_stats_after_exclusion['Stimuli Set 2 Participants'] = stimset_counts['stimuli_set2']
descriptive_stats_after_exclusion['Stimuli Set 3 Participants'] = stimset_counts['stimuli_set3']
print(f"There were {stimset_counts['stimuli_set1']} Participants tested with stimuli set 1, "
      f"{stimset_counts['stimuli_set2']} with stimuli set 2, and "
      f"{stimset_counts['stimuli_set3']} with stimuli set 3.")

# %% Save the descriptive statistics in a CSV file
descriptive_stats_after_exclusion_df = pd.DataFrame(list(descriptive_stats_after_exclusion.items()), columns=['Measure', 'Value'])
descriptive_stats_after_exclusion_df.to_csv('../data/csv_files/descriptive_statistics_after_exclusion.csv', index=False)