"""
Dataset endpoints for Email Spam Detection API
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json

router = APIRouter(prefix="/dataset")

@router.get("/info")
async def get_dataset_info():
    """
    Get dataset information and statistics
    """
    try:
        # Path to dataset info
        processed_dir = Path("data/processed")
        info_file = processed_dir / "dataset_info.json"
        
        if not info_file.exists():
            raise HTTPException(
                status_code=404, 
                detail="Dataset information not found. Preprocess the data first."
            )
        
        with open(info_file, 'r') as f:
            dataset_info = json.load(f)
        
        return JSONResponse(content=dataset_info)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset info: {str(e)}")

@router.get("/samples")
async def get_dataset_samples(
    split: str = "train",
    n_samples: int = 10,
    include_labels: bool = True
):
    """
    Get sample texts from the dataset
    """
    try:
        # Path to dataset splits
        processed_dir = Path("data/processed")
        split_file = processed_dir / f"{split}.csv"
        
        if not split_file.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Split '{split}' not found. Available splits: train, val, test"
            )
        
        # Read samples
        df = pd.read_csv(split_file)
        
        if n_samples > len(df):
            n_samples = len(df)
        
        samples = df.sample(n=min(n_samples, 100), random_state=42)
        
        # Prepare response
        response_samples = []
        for _, row in samples.iterrows():
            sample = {
                'text': row.get('text', '')[:200] + "..." if len(row.get('text', '')) > 200 else row.get('text', ''),
                'text_length': len(row.get('text', '')),
                'word_count': len(str(row.get('text', '')).split())
            }
            
            if include_labels and 'spam' in row:
                sample['label'] = int(row['spam'])
                sample['class'] = 'SPAM' if row['spam'] == 1 else 'HAM'
            
            response_samples.append(sample)
        
        return JSONResponse(content={
            'split': split,
            'total_samples': len(df),
            'returned_samples': len(response_samples),
            'samples': response_samples,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset samples: {str(e)}")

@router.get("/statistics")
async def get_dataset_statistics():
    """
    Get detailed dataset statistics
    """
    try:
        # Load all splits
        splits = ['train', 'val', 'test']
        statistics = {}
        
        for split in splits:
            split_file = Path(f"data/processed/{split}.csv")
            if split_file.exists():
                df = pd.read_csv(split_file)
                
                if 'spam' in df.columns:
                    spam_count = df['spam'].sum()
                    ham_count = len(df) - spam_count
                    
                    statistics[split] = {
                        'total_samples': len(df),
                        'spam_count': int(spam_count),
                        'ham_count': int(ham_count),
                        'spam_percentage': float((spam_count / len(df)) * 100) if len(df) > 0 else 0,
                        'text_statistics': {
                            'avg_length': float(df['text'].apply(len).mean()) if 'text' in df.columns else 0,
                            'avg_word_count': float(df['text'].apply(lambda x: len(str(x).split())).mean()) if 'text' in df.columns else 0,
                            'min_length': int(df['text'].apply(len).min()) if 'text' in df.columns else 0,
                            'max_length': int(df['text'].apply(len).max()) if 'text' in df.columns else 0
                        }
                    }
        
        # Overall statistics
        if statistics:
            total_samples = sum(stats['total_samples'] for stats in statistics.values())
            total_spam = sum(stats['spam_count'] for stats in statistics.values())
            
            overall_stats = {
                'total_samples': total_samples,
                'total_spam': total_spam,
                'total_ham': total_samples - total_spam,
                'overall_spam_percentage': float((total_spam / total_samples) * 100) if total_samples > 0 else 0
            }
        else:
            overall_stats = {}
        
        return JSONResponse(content={
            'splits': statistics,
            'overall': overall_stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset statistics: {str(e)}")

@router.get("/download/{split}")
async def download_dataset_split(split: str):
    """
    Download a dataset split as CSV
    """
    try:
        split_file = Path(f"data/processed/{split}.csv")
        
        if not split_file.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Split '{split}' not found. Available splits: train, val, test"
            )
        
        return FileResponse(
            split_file,
            media_type="text/csv",
            filename=f"email_spam_{split}.csv"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download dataset: {str(e)}")

@router.get("/visualization/{viz_type}")
async def get_dataset_visualization(viz_type: str):
    """
    Get dataset visualization plots
    """
    try:
        plots_dir = Path("outputs/plots")
        
        # Map visualization types to file names
        viz_files = {
            'class_distribution': 'class_distribution.png',
            'text_length_distribution': 'text_length_distribution.png',
            'word_count_distribution': 'word_count_distribution.png',
            'feature_distribution': 'feature_distribution_tfidf.png',
            'wordcloud': 'wordcloud_all.png'
        }
        
        if viz_type not in viz_files:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid visualization type. Available: {list(viz_files.keys())}"
            )
        
        file_path = plots_dir / viz_files[viz_type]
        
        if file_path.exists():
            return FileResponse(
                file_path,
                media_type="image/png",
                filename=f"dataset_{viz_type}.png"
            )
        else:
            # Try to generate the visualization
            raise HTTPException(
                status_code=404, 
                detail=f"Visualization not found. Generate it first by preprocessing/training the model."
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get visualization: {str(e)}")