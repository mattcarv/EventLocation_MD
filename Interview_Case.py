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

# count = df['Contact Type'].value_counts()
# count.plot(kind='bar')import numpy as np
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

combined = df_top.pivot_table(
    index='Contact Subtype',
    columns=['FSA','Client Flag'],
    aggfunc='size',
    fill_value=0
)

ratio = combined.xs(1, level=1, axis=1) \
        / (combined.xs(0, level=1, axis=1) + combined.xs(1, level=1, axis=1))

desired_order = ['Student', 'Resident', 'Specialist', 'GP/FM']

ratio = ratio.reindex(desired_order).dropna(how='all')

plt.figure(figsize=(30, max(6, 0.4 * len(ratio))))
plt.imshow(ratio, aspect='auto', cmap='plasma')
plt.colorbar(label='Current / Total (%)')

plt.xticks(range(ratio.shape[1]), ratio.columns, rotation=45, ha='right')
plt.yticks(range(ratio.shape[0]), ratio.index)

plt.title('Proportion of Current Clients by Career Stage in Top FSAs')

plt.show()
#%% How do we calculate a valuable client?




# plt.xlabel('Contact Type')
# plt.ylabel('Count')
#plt.show()

#----------------------------------------------------#

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

#%%
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
        currents     = ('Client Flag', 'sum'),
        prospects    = ('Client Flag', lambda x: (x == 0).sum()),
        avg_stage    = ('stage_score', 'mean'),
        total_clients= ('Client Flag', 'count')
    )
    .reset_index()
)

grouped['pct_current']   = grouped['currents']   / grouped['total_clients']
grouped['pct_prospects'] = grouped['prospects'] / grouped['total_clients']

min_size = 20
grouped = grouped[grouped['total_clients'] >= min_size]

plt.figure(figsize=(19, 8))

bubble_sizes = grouped['avg_stage']**2 * 200

sc = plt.scatter(
    grouped['pct_prospects'] * 100,   # as % on X
    grouped['pct_current']  * 100,    # as % on Y
    s=bubble_sizes,
    c=grouped['avg_stage'],
    cmap='viridis',
    alpha=0.8,
    edgecolors='w'
)
plt.colorbar(sc, label='Avg. Career Stage')

plt.axhline(50, color='gray', linestyle='--', linewidth=1)
plt.axvline(50, color='gray', linestyle='--', linewidth=1)

plt.xlabel('Prospective Clients (% of FSA)')
plt.ylabel('Current Clients     (% of FSA)')
plt.title('Client Mix by FSA\n(Bubble ∝ Avg. Career Stage)')

for _, row in grouped.iterrows():
    plt.text(
        row['pct_prospects']*100,
        row['pct_current']*100,
        row['FSA'],
        fontsize=8,
        ha='center',
        va='center'
    )

plt.tight_layout()
plt.show()
#%% Undestanding where we could find more prospective clients for their career 
#   stage

fsa_totals = (
    df
    .groupby('FSA')['Client Flag']
    .count()
    .reset_index(name='total')
)
top = fsa_totals.nlargest(30, 'total')['FSA']

df_top = df[df['FSA'].isin(top)]

combined = df_top.pivot_table(
    index='Contact Subtype',
    columns=['FSA','Client Flag'],
    aggfunc='size',
    fill_value=0
)

ratio = combined.xs(1, level=1, axis=1) \
        / (combined.xs(0, level=1, axis=1) + combined.xs(1, level=1, axis=1))

plt.figure(figsize=(30, max(6,0.4*len(ratio))))
plt.imshow(ratio, aspect='auto')
plt.colorbar(label='Current / Total (%)')
plt.xticks(range(ratio.shape[1]), ratio.columns, rotation=45, ha='right')
plt.yticks(range(ratio.shape[0]), ratio.index)
plt.title('Proportion of Current Clients by Specialty in Top FSAs')
plt.tight_layout()
plt.show()
