import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams.update({'font.size': 18})
#----------------------------------------------------#

df = pd.read_excel("C:/Users/mdoca/Downloads/Case_Study_Data.xlsx", 
                   usecols=[1, 2, 3, 4, 6])

#%%----------------------------------------------------#

count = df['Contact Type'].value_counts()
# count.plot(kind='bar')
# plt.xlabel('Contact Type')
# plt.ylabel('Count')
#plt.show()

counts = df.groupby(["Contact Type", "Client Flag"]).size().unstack(fill_value=0)

percentages = counts.div(counts.sum(axis=1), axis=0) * 100

# Plot
ax = percentages.plot(kind="bar", stacked=True, figsize=(18, 12))
plt.ylabel("Percentage")
plt.title("Client Flag Distribution by Contact Type")
plt.legend(title="Client Flag", labels=["Prospect", "Current"])

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', label_type='center')

plt.show()

# As expected, the proportion of client/prospect changes with stage in career
# for the doctors.

#%%

stage_mapping = {
    'Student':   1,
    'Resident':  2,
    'Practicing':3,
    'Retired':   4
}
df['stage_score'] = df['Contact Type'].map(stage_mapping)


grouped = (
    df
    .groupby('FSA')
    .agg(
        currents = ('Client Flag', 'sum'),
        prospects= ('Client Flag', lambda x: (x == 0).sum()),
        avg_stage = ('stage_score', 'mean')
    )
    .reset_index()
)

plt.figure(figsize=(20, 12))
bubble_sizes = (grouped['avg_stage']**2) * 100  # scale up for visibility

plt.scatter(
    grouped['prospects'], grouped['currents'],
    s=bubble_sizes, c=grouped['avg_stage'],
    cmap='viridis', alpha=0.8, edgecolors='w'
)
plt.colorbar(label='Avg. Career Stage')

plt.axhline(100, color='gray', linestyle='--', linewidth=1)
plt.axvline(100, color='gray', linestyle='--', linewidth=1)

plt.xlabel('Number of Prospective Clients')
plt.ylabel('Number of Current Clients')
plt.title('Prospects vs. Current Clients by FSA\n(Bubble size ∝ Avg. Career Stage)')

for _, row in grouped.iterrows():
    plt.text(
        row['prospects'],
        row['currents'],
        row['FSA'],
        fontsize=8,
        ha='center',
        va='center'
    )


plt.show()


#%% Understanding where we could find more prospective clients for their career 
#   stage

fsa_totals = (
    df
    .groupby('FSA')['Client Flag']
    .count()
    .reset_index(name='total')
)
top = fsa_totals.nlargest(30, 'total')['FSA']

df_top = df[df['FSA'].isin(top)]

# top_counts = fsa_totals[fsa_totals['FSA'].isin(top)] \
                # .sort_values('total', ascending=False) \
               #  .reset_index(drop=True)

# print("Top 30 FSAs by total MDs:")
# print(top_counts.to_string(index=False))
#%%
combined = df_top.pivot_table(
    index='Contact Subtype',
    columns=['FSA','Client Flag'],
    aggfunc='size',
    fill_value=0
)

ratio = combined.xs(1, level=1, axis=1) \
        / (combined.xs(0, level=1, axis=1) + combined.xs(1, level=1, axis=1))

order = ['Student', 'Resident', 'Specialist', 'GP/FM']

ratio = ratio.reindex(order).dropna(how='all')

plt.figure(figsize=(30, max(6, 0.4 * len(ratio))))
plt.imshow(ratio, aspect='auto', cmap='plasma')
plt.colorbar(label='Current / Total (%)')

plt.xticks(range(ratio.shape[1]), ratio.columns, rotation=45, ha='right')
plt.yticks(range(ratio.shape[0]), ratio.index)

plt.title('Proportion of Current Clients by Career Stage in Top FSAs')

plt.show()
#%% How do we calculate a valuable client?

# Introducing income to find which regions would have a higher monetary value.

# On average, Specialists bring more money, so where can we find prospects?

spec_counts = (
    df[df['FSA'].isin(['M5P','M2N','L9H','M4V','N6H'])]
    .groupby(['FSA','Contact Subtype'])
    .size()
    .unstack(fill_value=0)
)

spec_counts['Total_MDs'] = spec_counts.sum(axis=1)

spec_counts = spec_counts.sort_values('Total_MDs', ascending=False)

print(spec_counts)

# We have three distinct regions, one in the GTA (M2N, M5P and M4V). One in Hamilton
# area (L9H). One in London area (N6H).

#%% Understanding the area surrounding the final picks. For the sake of 
#   studying accessibility to the area.

# For the GTA, neighbouring regions look like:
    
df_gta = df[df['FSA'].isin(['M5P','M2N','M4V', 'M5N', 'M4R', 'M5M', 'M2P'])]

combined = df_gta.pivot_table(
    index='Contact Subtype',
    columns=['FSA','Client Flag'],
    aggfunc='size',
    fill_value=0
)

ratio = combined.xs(1, level=1, axis=1) \
        / (combined.xs(0, level=1, axis=1) + combined.xs(1, level=1, axis=1))

order = ['Student', 'Resident', 'Specialist', 'GP/FM']

ratio = ratio.reindex(order).dropna(how='all')

plt.figure(figsize=(30, max(6, 0.4 * len(ratio))))
plt.imshow(ratio, aspect='auto', cmap='plasma')
plt.colorbar(label='Current / Total (%)')

plt.xticks(range(ratio.shape[1]), ratio.columns, rotation=45, ha='right')
plt.yticks(range(ratio.shape[0]), ratio.index)

plt.title('Proportion of Current Clients by Career Stage in GTA')

plt.show()

#%%

# For Hamilton:
    
df_gta = df[df['FSA'].isin(['M5P','M2N','M4V', 'M5N', 'M4R', 'M5M', 'M2P'])]

combined = df_gta.pivot_table(
    index='Contact Subtype',
    columns=['FSA','Client Flag'],
    aggfunc='size',
    fill_value=0
)

ratio = combined.xs(1, level=1, axis=1) \
        / (combined.xs(0, level=1, axis=1) + combined.xs(1, level=1, axis=1))

order = ['Student', 'Resident', 'Specialist', 'GP/FM']

ratio = ratio.reindex(order).dropna(how='all')

plt.figure(figsize=(30, max(6, 0.4 * len(ratio))))
plt.imshow(ratio, aspect='auto', cmap='plasma')
plt.colorbar(label='Current / Total (%)')

plt.xticks(range(ratio.shape[1]), ratio.columns, rotation=45, ha='right')
plt.yticks(range(ratio.shape[0]), ratio.index)

plt.title('Proportion of Current Clients by Career Stage in GTA')

plt.show()

#%%

# For London

df_lon = df[df['FSA'].isin(['N6H', 'N6A', 'N6C', 'N6P', 'N6K'])]

combined = df_lon.pivot_table(
    index='Contact Subtype',
    columns=['FSA','Client Flag'],
    aggfunc='size',
    fill_value=0
)

ratio = combined.xs(1, level=1, axis=1) \
        / (combined.xs(0, level=1, axis=1) + combined.xs(1, level=1, axis=1))

order = ['Student', 'Resident', 'Specialist', 'GP/FM']

ratio = ratio.reindex(order).dropna(how='all')

plt.figure(figsize=(30, max(6, 0.4 * len(ratio))))
plt.imshow(ratio, aspect='auto', cmap='plasma')
plt.colorbar(label='Current / Total (%)')

plt.xticks(range(ratio.shape[1]), ratio.columns, rotation=45, ha='right')
plt.yticks(range(ratio.shape[0]), ratio.index)

plt.title('Proportion of Current Clients by Career Stage in GTA')

plt.show()
