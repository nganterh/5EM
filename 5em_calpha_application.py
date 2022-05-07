import itertools
from tkinter import N
import numpy as np
import pandas as pd
import pyreadstat
import os
import re
import streamlit as st
from tqdm import tqdm
from typing import  Optional

st.title('5EM data')

uploaded_file = st.file_uploader('Upload a File')
folder_sav_files = [file for file in os.listdir(os.getcwd()) if r'.sav' in file]

if not uploaded_file and len(folder_sav_files) == 0:
    st.write('*No .sav file to use')
    st.stop()

elif uploaded_file:
    # To read file as bytes:
    with open(os.path.join(os.getcwd(), uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer())

    file = [file for file in os.listdir(os.getcwd()) if file == uploaded_file.name][0]
    st.write('**Using uploaded file:**', file)

else:
    file = folder_sav_files[0]
    st.write('**Using default file:**', file)

# File data loading as a Pandas DataFrame and metadata as a dictionary
df, meta = pyreadstat.read_sav(file)

#Thanks to https://gist.github.com/lauramar6261/7b6b3ede3cb5f7e23c75f0a93ffb869e
import numpy as np

def svar(X):
    n = float(len(X))
    svar=(sum([(x-np.mean(X))**2 for x in X]) / n)* n/(n-1.)
    return svar


def CronbachAlpha2(df, col_names):
    items_scores = [df[col].to_list() for col in col_names]
    itemvars = [svar(item) for item in items_scores]
    tscores = [0] * len(items_scores[0])
    
    for item in items_scores:
        for i in range(len(item)):
            tscores[i]+= item[i]
            
    nitems = len(items_scores)
    Calpha = nitems/(nitems-1.) * (1-sum(itemvars)/ svar(tscores))
    
    return Calpha


def corr_items(col_names):
    combs = [comb for comb in itertools.combinations(col_names, len(col_names)-1)]
    dict_combs = {col: list([comb for comb in combs if col not in comb][0])
                  for col in col_names}

    for col, cols in dict_combs.items():
        df_left = df[col].rename(col).to_frame()
        df_right = df[cols].mean(axis=1).rename('mean_others').to_frame()
        dict_combs[col] = pd.merge(df_left, df_right, left_index=True,
                          right_index=True).corr().loc[col, 'mean_others']

    return dict_combs

def gen_construct(df, construct, col_names, n_items: Optional[int] = None):   
    # Setting up our combinations to calculate the Cronbach Alpha from
    if n_items is None:
        combs = [itertools.combinations(col_names, i+1) for i in range(len(col_names)) if i >= 1]
        combs_list = [list(item) for sublist in combs for item in sublist]

    else:
        combs = itertools.combinations(col_names, n_items)
        combs_list = [list(comb) for comb in combs]

    list_dicts = []

    # filling our dictionary list and storing as df
    for comb in combs_list:
        list_dicts.append({**{'alpha': CronbachAlpha2(df, col_names=comb)},
                           **corr_items(comb), **{'construct': construct}})
    return list_dicts


# Getting columns and results list ready
cols = list(meta.column_names_to_labels.keys())
list_bases = [  re.search('([\D]+)\d+', col).group(1) for col in cols
                if re.search('([\D]+)\d+', col) != None]

# Construct and associated constuting columns             
dict_constructs = { base: re.findall(f',({base}\d+)', ','.join(cols)) 
                    for base in set(list_bases) if len(re.findall(f',({base}\d+)',
                    ','.join(cols))) > 1}

# User input for constructs to be used
base = st.selectbox('Pick one', sorted(dict_constructs.keys()))

st.write(f"**Constructo:** {base}  \n" + 
f"**Valores:** {', '.join(list(dict_constructs[base]))}")


# Analyzing based on constructs' overlying columns
if len(dict_constructs[base]) <= 8:
    list_dfs = gen_construct(df, base, dict_constructs[base])
    df_cronbach = pd.DataFrame(list_dfs)
    
else:
    st.write('**Too many items to analyse**')
    constructs = list(dict_constructs[base])
    list_dfs = []

    for n_items in reversed(range(2, len(constructs)+1)):
        # Creating and sorting our DataFrame for each number of items
        df_local = pd.DataFrame(gen_construct(df, base, constructs, n_items))
        df_local.sort_values(by='alpha', ascending=False, inplace=True)

        # Adding to our list of DataFrames whilst extracting the worst alphas
        cols_numeric = [col for col in df_local.columns if col not in ['alpha', 'construct']]
        constructs.remove(df_local[cols_numeric].iloc[0].idxmin())
        list_dfs.append(df_local)

    df_cronbach = pd.concat(list_dfs)

df_cronbach.sort_values(by='alpha', ascending=False, inplace=True)
df_cronbach.drop(columns=['construct'], inplace=True)
df_cronbach.reset_index(drop=True, inplace=True)

st.dataframe(df_cronbach.style.format(precision=4))