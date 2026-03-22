import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import MinMaxScaler
from thefuzz import process, fuzz
import warnings

warnings.filterwarnings('ignore')

DATA_FOLDER = 'data'
INPUT_LAPTOP = os.path.join(DATA_FOLDER, 'laptop_price.csv')
INPUT_GPU_G3D = os.path.join(DATA_FOLDER, 'GPU_benchmarks_ALL_YEARS.csv')
INPUT_GPU_TDP = os.path.join(DATA_FOLDER, 'gpu_1986-2026.csv')
OUTPUT_FILE = os.path.join(DATA_FOLDER, 'laptop_processed.csv')


if not os.path.exists(INPUT_LAPTOP):
    print("Missing laptop_price.csv")
    exit()

try:
    laptop_df = pd.read_csv(INPUT_LAPTOP, encoding='latin-1')
except:
    laptop_df = pd.read_csv(INPUT_LAPTOP)

g3d_df = pd.read_csv(INPUT_GPU_G3D)
tdp_df = pd.read_csv(INPUT_GPU_TDP)

print(f"Laptop: {len(laptop_df)}")
print(f"G3D: {len(g3d_df)}")
print(f"TDP: {len(tdp_df)}")

g3d_df = g3d_df[['gpuName', 'G3Dmark']].sort_values(
    'G3Dmark', ascending=False
).drop_duplicates('gpuName', keep='first')

tdp_df['Full_Name'] = tdp_df['Brand'].astype(str) + ' ' + tdp_df['Name'].astype(str)

def clean_tdp(val):
    #Range ước lượng:
    #Ultrabook: 10-28W
    #Gaming mid: 45-90W
    #Gaming high-end: 100-175W
    #Gaming extreme/Workstation: 180-300W
    #Desktop: 350W+ (reject)

    if pd.isna(val): return np.nan
    match = re.search(r'(\d+)', str(val))
    if match:
        tdp = float(match.group(1))
        if 10 <= tdp <= 300:
            return tdp
        if tdp > 300:
            pass  
    return np.nan

tdp_df['Clean_TDP'] = tdp_df['Board Design__TDP'].apply(clean_tdp)
tdp_df_clean = tdp_df.dropna(subset=['Clean_TDP']).drop_duplicates('Full_Name')

print(f"G3D clean: {len(g3d_df)}")
print(f"TDP clean: {len(tdp_df_clean)}")

#Format: Company/Product/TypeName/Inches/ScreenResolution/Cpu
def create_model_string(row):
    components = [
        str(row.get('Company', '')),
        str(row.get('Product', '')),
        str(row.get('TypeName', '')),
        str(row.get('Inches', '')),
        str(row.get('ScreenResolution', '')),
        str(row.get('Cpu', ''))
    ]
    
    components = [c.strip() for c in components if c and c != 'nan' and c.strip()]
    
    return '/'.join(components)

laptop_df['Model'] = laptop_df.apply(create_model_string, axis=1)

print(f"{laptop_df['Model'].iloc[0][:80]}...")

laptop_df['Price_Original'] = laptop_df['Price_euros']
laptop_df['Price'] = pd.to_numeric(laptop_df['Price_euros'], errors='coerce').fillna(0)
print(f"Loại bỏ {(laptop_df['Price'] <= 0).sum()} dòng Price invalid")
laptop_df = laptop_df[laptop_df['Price'] > 0].copy()
laptop_df.reset_index(drop=True, inplace=True)

def clean_ram(text):
    if pd.isna(text): 
        return 4.0  
    
    text = str(text).upper()
    match = re.search(r'(\d+)\s*GB', text)  
    
    if match:
        val = float(match.group(1))
        if 2 <= val <= 128:
            return val
        return 4.0
    
    return 4.0

laptop_df['Ram_Original'] = laptop_df['Ram']
laptop_df['Ram'] = laptop_df['Ram'].apply(clean_ram)

print(f"RAM stats: min={laptop_df['Ram'].min()}, max={laptop_df['Ram'].max()}")

laptop_df['SSD_Original'] = laptop_df['Memory']

def calculate_storage_score(text):
    #SSD/Flash: 100% giá trị
    #Hybrid: 60% giá trị (SSD cache + HDD platter)
    #HDD: 20% giá trị (chậm)

    if pd.isna(text): 
        return 256.0 
    
    text = str(text).upper()
    
    pattern = r'(\d+(?:\.\d+)?)\s*(GB|TB)(?:\s+(SSD|HDD|FLASH|HYBRID))?'
    
    matches = re.findall(pattern, text)
    
    if not matches:
        return 256.0  
    
    total_score = 0.0
    
    for amount, unit, disk_type in matches:
        capacity = float(amount)
        if unit == 'TB':
            capacity *= 1024
        
        if not disk_type:
            if capacity <= 512:
                disk_type = 'SSD'
            else:
                disk_type = 'HDD'
        
        if 'FLASH' in text:
            disk_type = 'SSD'  
        
        if disk_type in ['SSD', 'FLASH']:
            weight = 1.0
        elif disk_type == 'HYBRID':
            weight = 0.6  
        else:  
            weight = 0.2
        
        total_score += capacity * weight
    
    return total_score

laptop_df['SSD'] = laptop_df['SSD_Original'].apply(calculate_storage_score)

print("\n 5 mẫu SS:")
sample = laptop_df[['SSD_Original', 'SSD']].sample(min(5, len(laptop_df)))
for _, row in sample.iterrows():
    print(f" '{row['SSD_Original']}' → {row['SSD']:.0f} điểm")

if (laptop_df['SSD'] == 0).sum() > 0:
    print(f"{(laptop_df['SSD']==0).sum()} dòng có SSD=0")

print("\nLọc bỏ máy cấu hình yếu (RAM < 4GB hoặc SSD < 128)")
count_before = len(laptop_df)

#Giữ lại các máy có RAM >= 4 và điểm SSD >= 128
laptop_df = laptop_df[ (laptop_df['Ram'] >= 4) & (laptop_df['SSD'] >= 128)]
laptop_df.reset_index(drop=True, inplace=True)

print(f"Đã loại bỏ: {count_before - len(laptop_df)}.")
print(f"Còn lại: {len(laptop_df)}.")

#fuzzy matching

laptop_df['Graphics'] = laptop_df['Gpu']

def clean_gpu_str(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    
    stopwords = [
        'nvidia', 'amd', 'intel', 'geforce', 'radeon', 'graphics', 
        'gpu', 'dedicated', 'vga', 'max-q', 'laptop', 'notebook',
        'ti-boost', 'mobile', 'card'
    ]
    
    for w in stopwords:
        s = re.sub(r'\b' + re.escape(w) + r'\b', '', s)
    
    s = re.sub(r'\bgb\b', '', s)
    s = re.sub(r'[^\w\s]', '', s)
    
    return ' '.join(s.split())

g3d_choices = list(g3d_df['gpuName'])
g3d_cleaned = [clean_gpu_str(c) for c in g3d_choices]
g3d_dict = dict(zip(g3d_df['gpuName'], g3d_df['G3Dmark']))

tdp_choices = list(tdp_df_clean['Full_Name'])
tdp_cleaned = [clean_gpu_str(c) for c in tdp_choices]
tdp_dict = dict(zip(tdp_df_clean['Full_Name'], tdp_df_clean['Clean_TDP']))

def batch_match_optimized(queries, cleaned_choices, original_choices, value_dict, threshold=70):
    results = []
    cache = {}
    
    for q in queries:
        clean_q = clean_gpu_str(q)
        
        if not clean_q:
            results.append(None)
            continue
        
        if clean_q in cache:
            results.append(cache[clean_q])
            continue
        
        match = process.extractOne(clean_q, cleaned_choices, scorer=fuzz.token_set_ratio)
        
        if match and match[1] >= threshold:
            idx = cleaned_choices.index(match[0])
            # Lấy tên gốc
            original_name = original_choices[idx]
            # Lấy giá trị từ dict
            value = value_dict.get(original_name)
            
            cache[clean_q] = value
            results.append(value)
        else:
            cache[clean_q] = None
            results.append(None)
    
    return results

#Match G3D
laptop_df['G3Dmark'] = batch_match_optimized(
    laptop_df['Graphics'], 
    g3d_cleaned, 
    g3d_choices, 
    g3d_dict,
    threshold=70
)

g3d_match = laptop_df['G3Dmark'].notna().sum()
g3d_rate = g3d_match / len(laptop_df) * 100
print(f"Match: {g3d_match}/{len(laptop_df)} ({g3d_rate:.1f}%)")

#Match TDP
laptop_df['TDP'] = batch_match_optimized(
    laptop_df['Graphics'],
    tdp_cleaned,
    tdp_choices,
    tdp_dict,
    threshold=75
)

tdp_match = laptop_df['TDP'].notna().sum()
tdp_rate = tdp_match / len(laptop_df) * 100
print(f"Match: {tdp_match}/{len(laptop_df)} ({tdp_rate:.1f}%)")

laptop_df['GPU'] = laptop_df['Graphics']

#Imputation
LOW_EUR, HIGH_EUR = 600, 1500

def get_segment(price):
    if pd.isna(price) or price <= 0:
        return 'unknown' 
    
    if price < LOW_EUR: 
        return 'low'
    elif price < HIGH_EUR: 
        return 'mid'
    return 'high'

laptop_df['segment'] = laptop_df['Price'].apply(get_segment)

medians_g3d = laptop_df.groupby('segment')['G3Dmark'].median().to_dict()
medians_tdp = laptop_df.groupby('segment')['TDP'].median().to_dict()

DEFAULTS_TDP = {'low': 15.0, 'mid': 45.0, 'high': 90.0, 'unknown': 45.0}
DEFAULTS_G3D = {'low': 1500, 'mid': 5000, 'high': 12000, 'unknown': 5000}

print(f"Median G3D: {medians_g3d}")
print(f"Median TDP: {medians_tdp}")

def vectorized_fill(series, segment_series, medians, defaults):
    segment_values = segment_series.map(medians)
    
    for seg, default in defaults.items():
        mask = (segment_series == seg) & segment_values.isna()
        segment_values[mask] = default
    
    return series.fillna(segment_values)

laptop_df['G3Dmark'] = vectorized_fill(
    laptop_df['G3Dmark'], 
    laptop_df['segment'], 
    medians_g3d, 
    DEFAULTS_G3D
)

laptop_df['TDP'] = vectorized_fill(
    laptop_df['TDP'],
    laptop_df['segment'],
    medians_tdp,
    DEFAULTS_TDP
)

laptop_df['Perf_Per_Watt'] = laptop_df['G3Dmark'] / (laptop_df['TDP'] + 1)

features = ['G3Dmark', 'Ram', 'SSD', 'TDP', 'Price', 'Perf_Per_Watt']
norm_names = ['Norm_G3Dmark', 'Norm_RAM', 'Norm_SSD', 'Norm_TDP', 'Norm_Price', 'Norm_Efficiency']

scaler = MinMaxScaler()
normalized = scaler.fit_transform(laptop_df[features])

for i, col in enumerate(norm_names):
    laptop_df[col] = normalized[:, i]


print("1. MAXIMIZE Performance")
print("2. MINIMIZE Energy Consumption (TDP)")
print("3. MINIMIZE Price")

laptop_df['Performance_Score'] = (
    0.50 * laptop_df['Norm_G3Dmark'] +
    0.25 * laptop_df['Norm_RAM'] +
    0.25 * laptop_df['Norm_SSD']
)

laptop_df['Energy_Consumption'] = laptop_df['Norm_TDP']

laptop_df['Price_Objective'] = laptop_df['Norm_Price']

laptop_df['Efficiency_Score'] = laptop_df['Norm_Efficiency']  
laptop_df['Budget_Score'] = 1 - laptop_df['Norm_Price']       

print(f"\nPerformance: [{laptop_df['Performance_Score'].min():.3f}, "
      f"{laptop_df['Performance_Score'].max():.3f}]")
print(f"Energy (TDP): [{laptop_df['Energy_Consumption'].min():.3f}, "
      f"{laptop_df['Energy_Consumption'].max():.3f}]")
print(f"Price: [{laptop_df['Price_Objective'].min():.3f}, "
      f"{laptop_df['Price_Objective'].max():.3f}]")


validation = {
    'Performance_Score có NaN': laptop_df['Performance_Score'].isna().sum(),
    'Energy_Consumption có NaN': laptop_df['Energy_Consumption'].isna().sum(),
    'Price_Objective có NaN': laptop_df['Price_Objective'].isna().sum(),
    'G3Dmark = 0': (laptop_df['G3Dmark'] == 0).sum(),
    'TDP = 0': (laptop_df['TDP'] == 0).sum(),
    'SSD = 0': (laptop_df['SSD'] == 0).sum(),
}

for key, val in validation.items():
    status = "OK" if val == 0 else "Error"
    print(f"{status} {key}: {val}")

EUR_TO_VND = 27000

laptop_df['Price_VND'] = laptop_df['Price'] * EUR_TO_VND

print(f"Tỷ giá: 1 EUR = {EUR_TO_VND:,} VND")
print(f"Giá VND: min={laptop_df['Price_VND'].min():,.0f}, max={laptop_df['Price_VND'].max():,.0f}")

output_cols = [
    'Model',
    'Price_Original',
    'Price',
    'Price_VND',
    'Ram_Original',
    'Ram',
    'SSD_Original',
    'SSD',
    'Graphics',
    'GPU',
    'G3Dmark',
    'TDP',
    'Perf_Per_Watt',
    'Price_Objective',
    'Budget_Score',
    'Performance_Score',
    'Energy_Consumption',
    'Efficiency_Score'
]

final_df = laptop_df[output_cols]
final_df.to_csv(OUTPUT_FILE, index=False)

print(f"DONE: {OUTPUT_FILE}")
print(f"Tổng: {len(final_df)} dòng")
print(f"Các cột: {len(output_cols)} cột\n")

print("preview")
for i in range(min(3, len(final_df))):
    print(f"      {i+1}. {final_df['Model'].iloc[i][:100]}")

print("\nTop 5 Performance:")
top5 = final_df.nlargest(5, 'Performance_Score')[
    ['Model', 'Price_VND', 'Performance_Score', 'Energy_Consumption']
]
for idx, (_, row) in enumerate(top5.iterrows(), 1):
    model_short = row['Model'][:40] if len(row['Model']) > 40 else row['Model']
    print(f"      {idx}. {model_short:40s} | "
          f"₫{row['Price_VND']:>12,.0f} | "
          f"P:{row['Performance_Score']:.3f} E:{row['Energy_Consumption']:.3f}")

print("\nTop 5 Budget:")
top5_budget = final_df.nlargest(5, 'Budget_Score')[
    ['Model', 'Price_VND', 'Budget_Score', 'Performance_Score']
]
for idx, (_, row) in enumerate(top5_budget.iterrows(), 1):
    model_short = row['Model'][:40] if len(row['Model']) > 40 else row['Model']
    print(f"      {idx}. {model_short:40s} | "
          f"₫{row['Price_VND']:>12,.0f} | "
          f"Budget:{row['Budget_Score']:.3f} Perf:{row['Performance_Score']:.3f}")

