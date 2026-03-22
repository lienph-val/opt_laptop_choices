import pandas as pd
import numpy as np 
import requests
import io
import os
import re

URL = "https://www.videocardbenchmark.net/gpu_list.php"
OUTPUT_FILE = os.path.join('data', 'GPU_benchmarks_ALL_YEARS.csv')

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

try:
    response = requests.get(URL, headers=headers)
    response.raise_for_status()
    print("OK")
except Exception as e:
    print(f"Error: {e}")
    exit()

try:
    dfs = pd.read_html(io.StringIO(response.text))
    gpu_df = max(dfs, key=len)
    print(f"Done: {len(gpu_df)} GPU.")

    
except Exception as e:
    print(f"Error {e}")
    exit()

def find_col(df, keywords):
    for col in df.columns:
        for kw in keywords:
            if kw.lower() in str(col).lower(): return col
    return None

col_name = find_col(gpu_df, ['Name', 'Videocard'])
col_g3d = find_col(gpu_df, ['G3D', 'Passmark', 'Rating'])
col_tdp = find_col(gpu_df, ['TDP', 'Power', 'Watt']) 
col_cat = find_col(gpu_df, ['Category', 'Type'])

print(f"   Mapping: Name='{col_name}' | G3D='{col_g3d}' | TDP='{col_tdp}'")

rename_map = {
    col_name: 'gpuName',
    col_g3d: 'G3Dmark',
    col_tdp: 'TDP',
    col_cat: 'category'
}
gpu_df = gpu_df.rename(columns=rename_map)

if 'TDP' not in gpu_df.columns:
    gpu_df['TDP'] = np.nan 
if 'category' not in gpu_df.columns:
    gpu_df['category'] = 'Unknown'

final_df = gpu_df[['gpuName', 'G3Dmark', 'TDP', 'category']].copy()

def clean_num(val):
    s = str(val).replace(',', '').replace('*', '').strip()
    if s.lower() in ['na', 'n/a', '-', 'nan', 'none']: return np.nan
    try:
        match = re.search(r'(\d+(\.\d+)?)', s)
        if match: return float(match.group(1))
    except: pass
    return np.nan

final_df['G3Dmark'] = final_df['G3Dmark'].apply(clean_num)
final_df['TDP'] = final_df['TDP'].apply(clean_num)
final_df['category'] = final_df['category'].fillna('Unknown')

if not os.path.exists('data'): os.makedirs('data')
final_df.to_csv(OUTPUT_FILE, index=False)

print("DONE")