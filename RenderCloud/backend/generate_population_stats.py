"""
生成人群统计数据的脚本
从 NHANES 数据集中计算每个疾病、年龄段、性别的真实患病率（得病率）
患病率 = 患病人数 / 总人数
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据集路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT.parent.parent / "dataset" / "nhanes_2021_2023_master.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "population_stats.json"


def get_age_range(age):
    """将年龄转换为年龄段"""
    if pd.isna(age):
        return None
    age = int(age)
    if age < 20:
        return "under_20"
    elif age < 30:
        return "20_30"
    elif age < 40:
        return "30_40"
    elif age < 50:
        return "40_50"
    elif age < 60:
        return "50_60"
    elif age < 70:
        return "60_70"
    elif age < 80:
        return "70_80"
    else:
        return "80_plus"


def calculate_disease_rate(df_group, label_col):
    """
    计算患病率
    
    Parameters:
    df_group: 分组后的数据框
    label_col: 标签列名
    
    Returns:
    tuple: (患病率百分比, 总人数, 患病人数)
    """
    if label_col not in df_group.columns:
        return None, 0, 0
    
    # 过滤掉标签为空的数据，以及不确定的值（3, 9等）
    # 只保留明确回答 Yes(1) 或 No(2) 的数据
    valid_df = df_group[
        (df_group[label_col].notna()) & 
        (df_group[label_col].isin([1, 2]))
    ]
    
    if len(valid_df) == 0:
        return None, 0, 0
    
    # 计算患病率：1=Yes (患病), 2=No (不患病)
    total_count = len(valid_df)
    disease_count = len(valid_df[valid_df[label_col] == 1])
    
    if total_count == 0:
        return None, 0, 0
    
    disease_rate = (disease_count / total_count) * 100  # 转换为百分比
    
    return disease_rate, total_count, disease_count


def generate_population_stats():
    """生成人群统计数据 - 基于真实的患病率"""
    # 检查数据集文件
    if not DATASET_PATH.exists():
        logger.error(f"数据集文件不存在: {DATASET_PATH}")
        return None
    
    logger.info(f"正在加载数据: {DATASET_PATH}")
    try:
        df = pd.read_csv(DATASET_PATH, low_memory=False)
        logger.info(f"数据加载完成，共 {len(df)} 行，{len(df.columns)} 列")
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return None
    
    # 定义疾病的标签字段
    # 1 = Yes (患病), 2 = No (不患病), 9 = 不确定/拒绝回答
    disease_labels = {
        'diabetes': 'DIQ010',        # 糖尿病病史 (1=Yes, 2=No, 3=不确定)
        'hypertension': 'BPQ020',    # 高血压病史 (1=Yes, 2=No)
        'cvd': 'MCQ160B',            # 心脏病 (1=Yes, 2=No)
        'ckd': 'MCQ220'              # 肾病病史 (1=Yes, 2=No)
    }
    
    # 检查标签字段是否存在
    for disease, label_col in disease_labels.items():
        if label_col and label_col not in df.columns:
            logger.warning(f"{disease} 的标签字段 {label_col} 不存在于数据集中")
    
    # 存储统计结果
    stats_result = {}
    
    # 年龄段定义
    age_ranges = {
        "under_20": (0, 20),
        "20_30": (20, 30),
        "30_40": (30, 40),
        "40_50": (40, 50),
        "50_60": (50, 60),
        "60_70": (60, 70),
        "70_80": (70, 80),
        "80_plus": (80, 200)
    }
    
    # 处理每个疾病
    for disease, label_col in disease_labels.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"正在处理疾病: {disease}")
        logger.info(f"{'='*60}")
        
        if label_col is None:
            logger.warning(f"{disease} 没有标签字段，跳过")
            stats_result[disease] = {}
            continue
        
        if label_col not in df.columns:
            logger.warning(f"{disease} 的标签字段 {label_col} 不存在，跳过")
            stats_result[disease] = {}
            continue
        
        stats_result[disease] = {}
        
        # 按年龄段和性别分组统计
        for age_range_key, (age_min, age_max) in age_ranges.items():
            stats_result[disease][age_range_key] = {}
            
            for gender_name in ["male", "female"]:
                gender_code = 1 if gender_name == "male" else 2
                
                # 筛选该组的数据
                mask = (
                    (df['RIDAGEYR'] >= age_min) & 
                    (df['RIDAGEYR'] < age_max) & 
                    (df['RIAGENDR'] == gender_code)
                )
                group_df = df[mask]
                
                # 计算患病率
                disease_rate, total_count, disease_count = calculate_disease_rate(group_df, label_col)
                
                if disease_rate is not None and total_count > 0:
                    stats_result[disease][age_range_key][gender_name] = {
                        "mean": float(disease_rate),  # 患病率（百分比）
                        "std_dev": 0.0,  # 患病率是比例，不需要标准差
                        "sample_size": int(total_count)
                    }
                    logger.info(f"  {age_range_key} {gender_name}: 患病率={disease_rate:.2f}%, 样本数={total_count}, 患病人数={disease_count}")
                else:
                    # 如果没有有效数据，返回0%患病率，样本数为0
                    stats_result[disease][age_range_key][gender_name] = {
                        "mean": 0.0,  # 患病率为0%
                        "std_dev": 0.0,
                        "sample_size": 0
                    }
                    logger.info(f"  {age_range_key} {gender_name}: 无有效数据，设置为0%")
        
        logger.info(f"{disease} 统计完成")
    
    # 保存结果
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(stats_result, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"统计结果已保存到: {OUTPUT_PATH}")
    logger.info(f"{'='*60}")
    
    return stats_result


if __name__ == "__main__":
    generate_population_stats()
