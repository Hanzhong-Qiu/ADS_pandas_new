
import pandas as pd
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

                                               
print("=" * 60)
print("1. 加载推文情感数据")
print("=" * 60)

df_tweets = pd.read_csv('/home/qqq/ADS_pandas_new/.csv/cleaned_sentiment_data.csv')
df_tweets['date'] = pd.to_datetime(df_tweets['date'])
print(f"推文数据: {len(df_tweets)} 天")
print(f"  范围: {df_tweets['date'].min().date()} 至 {df_tweets['date'].max().date()}")
print(f"  列: {list(df_tweets.columns)}")

                                               
print("\n" + "=" * 60)
print("2. 加载 OxCGRT 数据")
print("=" * 60)

df_ox = pd.read_csv('/home/qqq/ADS_pandas_new/.csv/OxCGRT_compact_national_v1.csv', low_memory=False)
df_ox['date'] = pd.to_datetime(df_ox['Date'], format='%Y%m%d')

print(f"OxCGRT 原始数据: {len(df_ox)} 行, {len(df_ox.columns)} 列")
print(f"  国家数: {df_ox['CountryName'].nunique()}")
print(f"  范围: {df_ox['date'].min().date()} 至 {df_ox['date'].max().date()}")

                           
print("\n可用的政策指标列:")
for col in sorted(df_ox.columns):
    if col.startswith(('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'H6',
                       'Stringency', 'Confirmed')):
                    
        non_null = df_ox[col].notna().sum()
        print(f"  {col:45s}  ({non_null} non-null)")

                                               
print("\n" + "=" * 60)
print("3. 筛选 Anglosphere 国家")
print("=" * 60)

COUNTRIES = ['United States', 'United Kingdom', 'Canada',
             'Australia', 'New Zealand', 'Ireland']

df_ang = df_ox[df_ox['CountryName'].isin(COUNTRIES)].copy()
print(f"筛选后: {len(df_ang)} 行")
print(f"包含国家: {sorted(df_ang['CountryName'].unique())}")

            
for c in COUNTRIES:
    n = len(df_ang[df_ang['CountryName'] == c])
    print(f"  {c}: {n} 天")

                                               
print("\n" + "=" * 60)
print("4. 提取指标并按日期取均值")
print("=" * 60)

                             
def find_col(df, keyword):
    matches = [c for c in df.columns if keyword in c]
    if matches:
        return matches[0]
    return None

       
sub_indicators = {
    'C1_School_closing':          find_col(df_ang, 'C1M_School'),
    'C2_Workplace_closing':       find_col(df_ang, 'C2M_Workplace'),
    'C4_Restrictions_gatherings': find_col(df_ang, 'C4M_Restrictions'),
    'C6_Stay_at_home':            find_col(df_ang, 'C6M_Stay'),
    'C7_Internal_movement':       find_col(df_ang, 'C7M_Restrictions'),
    'C8_International_travel':    find_col(df_ang, 'C8EV_International'),
    'H6_Facial_coverings':        find_col(df_ang, 'H6M_Facial'),
}

                
stringency_col = find_col(df_ang, 'StringencyIndex_Average')

      
cases_col = find_col(df_ang, 'ConfirmedCases')
deaths_col = find_col(df_ang, 'ConfirmedDeaths')

         
print("\n找到的列名映射:")
for name, col in sub_indicators.items():
    print(f"  {name:35s} → {col}")
print(f"  {'StringencyIndex':35s} → {stringency_col}")
print(f"  {'ConfirmedCases':35s} → {cases_col}")
print(f"  {'ConfirmedDeaths':35s} → {deaths_col}")

             
cols_to_extract = [c for c in [stringency_col, cases_col, deaths_col] if c]
cols_to_extract += [c for c in sub_indicators.values() if c]

                        
df_daily = df_ang.groupby('date')[cols_to_extract].mean().reset_index()
print(f"\n按日期聚合后: {len(df_daily)} 天")

            
rename_map = {}
if stringency_col:
    rename_map[stringency_col] = 'StringencyIndex_Average'
if cases_col:
    rename_map[cases_col] = 'ConfirmedCases'
if deaths_col:
    rename_map[deaths_col] = 'ConfirmedDeaths'
for new_name, old_name in sub_indicators.items():
    if old_name:
        rename_map[old_name] = new_name

df_daily = df_daily.rename(columns=rename_map)

                                               
print("\n" + "=" * 60)
print("5. 计算每日新增")
print("=" * 60)

df_daily = df_daily.sort_values('date').reset_index(drop=True)

if 'ConfirmedCases' in df_daily.columns:
    df_daily['daily_new_cases'] = df_daily['ConfirmedCases'].diff()
                            
    df_daily['daily_new_cases'] = df_daily['daily_new_cases'].clip(lower=0)
    print(f"  daily_new_cases: 均值={df_daily['daily_new_cases'].mean():.0f}")

if 'ConfirmedDeaths' in df_daily.columns:
    df_daily['daily_new_deaths'] = df_daily['ConfirmedDeaths'].diff()
    df_daily['daily_new_deaths'] = df_daily['daily_new_deaths'].clip(lower=0)
    print(f"  daily_new_deaths: 均值={df_daily['daily_new_deaths'].mean():.0f}")

                                               
print("\n" + "=" * 60)
print("6. 合并推文数据和OxCGRT数据")
print("=" * 60)

df_merged = pd.merge(df_tweets, df_daily, on='date', how='inner')
df_merged = df_merged.sort_values('date').reset_index(drop=True)

print(f"合并后: {len(df_merged)} 天")
print(f"  范围: {df_merged['date'].min().date()} 至 {df_merged['date'].max().date()}")
print(f"\n所有列:")
for col in df_merged.columns:
    non_null = df_merged[col].notna().sum()
    print(f"  {col:35s}  {non_null}/{len(df_merged)} non-null")

                                               
print("\n" + "=" * 60)
print("7. 保存")
print("=" * 60)

output_path = '/home/qqq/ADS_pandas_new/.csv/new/full_analysis_data.csv'
df_merged.to_csv(output_path, index=False)
print(f"✅ 完整分析数据已保存: {output_path}")
print(f"   {len(df_merged)} 行 × {len(df_merged.columns)} 列")

               
print(f"\n前3行预览:")
print(df_merged.head(3).to_string())

print(f"\n描述性统计:")
numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
print(df_merged[numeric_cols].describe().round(2).to_string())