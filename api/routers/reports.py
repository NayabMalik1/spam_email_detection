"""
Reports endpoints for Email Spam Detection API
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime
from pathlib import Path
import json

router = APIRouter(prefix="/reports")

@router.get("/list")
async def list_reports():
    """
    List all available reports
    """
    try:
        reports_dir = Path("outputs/reports")
        
        if not reports_dir.exists():
            return JSONResponse(content={
                'reports': [],
                'message': 'No reports directory found'
            })
        
        # Find all report files
        report_files = []
        for file_path in reports_dir.glob("*.json"):
            report_files.append({
                'name': file_path.name,
                'path': str(file_path),
                'size': file_path.stat().st_size,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                'type': file_path.stem.split('_')[0] if '_' in file_path.stem else 'other'
            })
        
        # Find all text reports
        text_reports = []
        for file_path in reports_dir.glob("*.txt"):
            text_reports.append({
                'name': file_path.name,
                'path': str(file_path),
                'size': file_path.stat().st_size,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
        
        # Group by type
        report_types = {}
        for report in report_files:
            report_type = report['type']
            if report_type not in report_types:
                report_types[report_type] = []
            report_types[report_type].append(report)
        
        return JSONResponse(content={
            'report_types': report_types,
            'text_reports': text_reports,
            'total_reports': len(report_files) + len(text_reports),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")

@router.get("/{report_name}")
async def get_report(report_name: str):
    """
    Get a specific report by name
    """
    try:
        reports_dir = Path("outputs/reports")
        report_path = reports_dir / report_name
        
        if not report_path.exists():
            # Try with .json extension
            report_path = reports_dir / f"{report_name}.json"
            if not report_path.exists():
                # Try with .txt extension
                report_path = reports_dir / f"{report_name}.txt"
                if not report_path.exists():
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Report '{report_name}' not found"
                    )
        
        # Check file type
        if report_path.suffix == '.json':
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            return JSONResponse(content={
                'report_name': report_path.name,
                'report_type': 'json',
                'data': report_data,
                'timestamp': datetime.now().isoformat()
            })
        
        elif report_path.suffix == '.txt':
            with open(report_path, 'r') as f:
                report_content = f.read()
            
            return JSONResponse(content={
                'report_name': report_path.name,
                'report_type': 'text',
                'content': report_content,
                'timestamp': datetime.now().isoformat()
            })
        
        else:
            # For other file types, return as file
            return FileResponse(
                report_path,
                filename=report_path.name
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get report: {str(e)}")

@router.get("/latest/{report_type}")
async def get_latest_report(report_type: str):
    """
    Get the latest report of a specific type
    """
    try:
        reports_dir = Path("outputs/reports")
        
        # Find all files of the specified type
        if report_type == 'evaluation':
            pattern = "evaluation_results_*.json"
        elif report_type == 'training':
            pattern = "training_report_*.json"
        elif report_type == 'threshold':
            pattern = "threshold_analysis_*.json"
        elif report_type == 'comparison':
            pattern = "comparison_results_*.json"
        elif report_type == 'error':
            pattern = "error_analysis_*.json"
        elif report_type == 'summary':
            pattern = "*_summary.txt"
        else:
            pattern = f"*{report_type}*.json"
        
        report_files = list(reports_dir.glob(pattern))
        
        if not report_files:
            raise HTTPException(
                status_code=404, 
                detail=f"No {report_type} reports found"
            )
        
        # Get latest file
        latest_file = max(report_files, key=lambda x: x.stat().st_mtime)
        
        # Read and return the report
        if latest_file.suffix == '.json':
            with open(latest_file, 'r') as f:
                report_data = json.load(f)
            
            return JSONResponse(content={
                'report_name': latest_file.name,
                'report_type': report_type,
                'data': report_data,
                'timestamp': datetime.now().isoformat()
            })
        
        elif latest_file.suffix == '.txt':
            with open(latest_file, 'r') as f:
                report_content = f.read()
            
            return JSONResponse(content={
                'report_name': latest_file.name,
                'report_type': report_type,
                'content': report_content,
                'timestamp': datetime.now().isoformat()
            })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get latest report: {str(e)}")

@router.get("/download/{report_name}")
async def download_report(report_name: str):
    """
    Download a report file
    """
    try:
        reports_dir = Path("outputs/reports")
        report_path = reports_dir / report_name
        
        if not report_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Report '{report_name}' not found"
            )
        
        return FileResponse(
            report_path,
            filename=report_path.name,
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download report: {str(e)}")

@router.get("/dashboard")
async def get_dashboard_data():
    """
    Get data for the web dashboard
    """
    try:
        # Get latest evaluation results
        reports_dir = Path("outputs/reports")
        eval_files = list(reports_dir.glob("evaluation_results_*.json"))
        
        if eval_files:
            latest_eval = max(eval_files, key=lambda x: x.stat().st_mtime)
            with open(latest_eval, 'r') as f:
                eval_data = json.load(f)
        else:
            eval_data = {}
        
        # Get prediction statistics from predictor
        # This would require access to the predictor instance
        # For now, use mock data
        prediction_stats = {
            'total_predictions': 1000,
            'spam_predictions': 350,
            'ham_predictions': 650,
            'avg_confidence': 0.85
        }
        
        # Get model info
        models_dir = Path("models/saved_models")
        config_files = list(models_dir.glob("*.json"))
        
        if config_files:
            latest_config = max(config_files, key=lambda x: x.stat().st_mtime)
            with open(latest_config, 'r') as f:
                model_config = json.load(f)
        else:
            model_config = {}
        
        # Get dataset info
        dataset_info = {}
        info_file = Path("data/processed/dataset_info.json")
        if info_file.exists():
            with open(info_file, 'r') as f:
                dataset_info = json.load(f)
        
        # Compile dashboard data
        dashboard_data = {
            'performance_metrics': {
                'accuracy': eval_data.get('metrics', {}).get('accuracy', 0),
                'precision': eval_data.get('metrics', {}).get('precision', 0),
                'recall': eval_data.get('metrics', {}).get('recall', 0),
                'f1_score': eval_data.get('metrics', {}).get('f1_score', 0),
                'roc_auc': eval_data.get('metrics', {}).get('roc_auc', 0)
            },
            'prediction_stats': prediction_stats,
            'model_info': {
                'name': model_config.get('model_name', 'Unknown'),
                'hidden_layers': model_config.get('hidden_layers', []),
                'accuracy': eval_data.get('metrics', {}).get('accuracy', 0)
            },
            'dataset_info': {
                'total_samples': dataset_info.get('total_samples', 0),
                'splits': dataset_info.get('splits', [])
            },
            'recent_activity': {
                'last_training': model_config.get('timestamp', ''),
                'last_evaluation': eval_data.get('timestamp', ''),
                'total_reports': len(list(reports_dir.glob("*.json"))) + len(list(reports_dir.glob("*.txt")))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")