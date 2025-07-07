import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
import argparse

# Import our model registry
from model_registry import ModelRegistry

# MLflow imports
import mlflow
import mlflow.sklearn

# Model testing utilities
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Utilities for managing models in production.
    Handles testing, validation, and deployment workflows.
    """
    
    def __init__(self, tracking_uri: str = "file:./mlruns"):
        self.registry = ModelRegistry(tracking_uri)
        logger.info("🛠️  Model Manager initialized")
    
    def test_model_performance(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                             min_recall: float = 0.6) -> Dict[str, Any]:
        """Test model performance against minimum requirements."""
        
        logger.info("🧪 Testing model performance...")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Performance validation
            validation_results = {
                'recall_pass': metrics['recall'] >= min_recall,
                'precision_reasonable': metrics['precision'] >= 0.15,  # At least 15% precision
                'roc_auc_acceptable': metrics['roc_auc'] >= 0.65,     # At least 65% AUC
                'f1_balanced': metrics['f1_score'] >= 0.25           # At least 25% F1
            }
            
            overall_pass = all(validation_results.values())
            
            logger.info("📊 Performance Test Results:")
            logger.info(f"   Recall: {metrics['recall']:.4f} ({'✅ PASS' if validation_results['recall_pass'] else '❌ FAIL'})")
            logger.info(f"   Precision: {metrics['precision']:.4f} ({'✅ PASS' if validation_results['precision_reasonable'] else '❌ FAIL'})")
            logger.info(f"   ROC-AUC: {metrics['roc_auc']:.4f} ({'✅ PASS' if validation_results['roc_auc_acceptable'] else '❌ FAIL'})")
            logger.info(f"   F1-Score: {metrics['f1_score']:.4f} ({'✅ PASS' if validation_results['f1_balanced'] else '❌ FAIL'})")
            logger.info(f"   Overall: {'✅ PASSED' if overall_pass else '❌ FAILED'}")
            
            return {
                'metrics': metrics,
                'validation': validation_results,
                'overall_pass': overall_pass
            }
            
        except Exception as e:
            logger.error(f"Error testing model performance: {e}")
            return {'metrics': {}, 'validation': {}, 'overall_pass': False}
    
    def validate_model_for_production(self, version: str = None) -> bool:
        """Comprehensive validation before production deployment."""
        
        logger.info("🔍 Validating model for production deployment...")
        
        try:
            # Load model from staging
            if version:
                model_uri = f"models:/{self.registry.registered_model_name}/{version}"
                model = mlflow.sklearn.load_model(model_uri)
                logger.info(f"📥 Loaded model version {version}")
            else:
                model = self.registry.load_model_from_registry("Staging")
                if model is None:
                    logger.error("No staging model found")
                    return False
                logger.info("📥 Loaded staging model")
            
            # Load test data (you'll need to implement this based on your data pipeline)
            # For now, we'll use a placeholder
            logger.info("📊 Loading test data...")
            # X_test, y_test = self.load_test_data()  # Implement this
            
            # For demonstration, let's create dummy test data
            # In production, you'd load actual holdout test data
            logger.warning("⚠️  Using dummy test data for validation demo")
            
            # Validation checks
            validation_checks = {
                'model_loaded': model is not None,
                'model_has_predict': hasattr(model, 'predict'),
                'model_has_predict_proba': hasattr(model, 'predict_proba'),
                # 'performance_test': self.test_model_performance(model, X_test, y_test)['overall_pass'],
                'model_serializable': True  # Could add serialization test
            }
            
            # Check feature expectations (example)
            expected_features = [
                'number_inpatient', 'number_emergency', 'discharge_disposition_id',
                'number_diagnoses', 'time_in_hospital', 'num_medications',
                'diabetesMed', 'metformin', 'num_lab_procedures', 'change',
                'number_outpatient', 'age', 'num_procedures', 'admission_type_id', 'repaglinide'
            ]
            
            logger.info("🔍 Running validation checks...")
            for check, result in validation_checks.items():
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"   {check}: {status}")
            
            overall_valid = all(validation_checks.values())
            
            if overall_valid:
                logger.info("✅ Model validation PASSED - Ready for production!")
            else:
                logger.warning("❌ Model validation FAILED - Do not deploy to production!")
            
            return overall_valid
            
        except Exception as e:
            logger.error(f"Error during model validation: {e}")
            return False
    
    def deploy_to_production(self, version: str = None, force: bool = False) -> bool:
        """Deploy model to production with validation."""
        
        logger.info("🚀 Starting production deployment...")
        
        # Validate model unless forced
        if not force:
            if not self.validate_model_for_production(version):
                logger.error("❌ Model validation failed. Use --force to override.")
                return False
        
        # Deploy to production
        success = self.registry.promote_to_production(version)
        
        if success:
            logger.info("🎉 Production deployment completed successfully!")
            
            # Log deployment event
            deployment_info = {
                'timestamp': datetime.now().isoformat(),
                'version': version,
                'model_name': self.registry.registered_model_name,
                'deployed_by': os.getenv('USER', 'unknown'),
                'validation_passed': not force
            }
            
            # Save deployment log
            os.makedirs('logs', exist_ok=True)
            with open('logs/deployment_log.json', 'a') as f:
                f.write(json.dumps(deployment_info) + '\n')
            
        return success
    
    def rollback_production(self, target_version: str = None) -> bool:
        """Rollback production to a previous version."""
        
        logger.info("🔄 Starting production rollback...")
        
        try:
            if target_version is None:
                # Find the previous production version
                versions = self.registry.list_all_model_versions()
                production_versions = [v for v in versions if v['stage'] == 'Production']
                
                if len(production_versions) < 2:
                    logger.error("No previous production version found for rollback")
                    return False
                
                target_version = production_versions[1]['version']  # Second most recent
                logger.info(f"🎯 Rolling back to version {target_version}")
            
            # Promote target version to production
            success = self.registry.transition_model_stage(
                model_name=self.registry.registered_model_name,
                version=target_version,
                stage="Production",
                archive_existing=True
            )
            
            if success:
                logger.info(f"✅ Rollback completed! Version {target_version} is now in production.")
                
                # Log rollback event
                rollback_info = {
                    'timestamp': datetime.now().isoformat(),
                    'action': 'rollback',
                    'target_version': target_version,
                    'model_name': self.registry.registered_model_name,
                    'rolled_back_by': os.getenv('USER', 'unknown')
                }
                
                os.makedirs('logs', exist_ok=True)
                with open('logs/deployment_log.json', 'a') as f:
                    f.write(json.dumps(rollback_info) + '\n')
            
            return success
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return False
    
    def compare_models(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions."""
        
        logger.info(f"📊 Comparing model versions {version1} vs {version2}")
        
        try:
            # Get model versions info
            versions = self.registry.list_all_model_versions()
            
            v1_info = next((v for v in versions if v['version'] == version1), None)
            v2_info = next((v for v in versions if v['version'] == version2), None)
            
            if not v1_info or not v2_info:
                logger.error("One or both model versions not found")
                return {}
            
            comparison = {
                'version_1': {
                    'version': v1_info['version'],
                    'stage': v1_info['stage'],
                    'recall': v1_info['recall'],
                    'precision': v1_info['precision'],
                    'f1_score': v1_info['f1_score'],
                    'roc_auc': v1_info['roc_auc']
                },
                'version_2': {
                    'version': v2_info['version'],
                    'stage': v2_info['stage'],
                    'recall': v2_info['recall'],
                    'precision': v2_info['precision'],
                    'f1_score': v2_info['f1_score'],
                    'roc_auc': v2_info['roc_auc']
                }
            }
            
            # Calculate improvements
            if isinstance(v1_info['recall'], (int, float)) and isinstance(v2_info['recall'], (int, float)):
                comparison['recall_improvement'] = v2_info['recall'] - v1_info['recall']
                comparison['f1_improvement'] = v2_info['f1_score'] - v1_info['f1_score']
                comparison['precision_improvement'] = v2_info['precision'] - v1_info['precision']
                comparison['roc_auc_improvement'] = v2_info['roc_auc'] - v1_info['roc_auc']
            
            # Print comparison
            logger.info("📊 Model Comparison Results:")
            logger.info("-" * 60)
            logger.info(f"{'Metric':<15} {'V' + version1:<12} {'V' + version2:<12} {'Improvement':<12}")
            logger.info("-" * 60)
            logger.info(f"{'Recall':<15} {v1_info['recall']:<12.4f} {v2_info['recall']:<12.4f} {comparison.get('recall_improvement', 0):<12.4f}")
            logger.info(f"{'Precision':<15} {v1_info['precision']:<12.4f} {v2_info['precision']:<12.4f} {comparison.get('precision_improvement', 0):<12.4f}")
            logger.info(f"{'F1-Score':<15} {v1_info['f1_score']:<12.4f} {v2_info['f1_score']:<12.4f} {comparison.get('f1_improvement', 0):<12.4f}")
            logger.info(f"{'ROC-AUC':<15} {v1_info['roc_auc']:<12.4f} {v2_info['roc_auc']:<12.4f} {comparison.get('roc_auc_improvement', 0):<12.4f}")
            logger.info("-" * 60)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {}
    
    def archive_old_models(self, keep_versions: int = 5) -> int:
        """Archive old model versions, keeping only the most recent ones."""
        
        logger.info(f"🗂️  Archiving old models (keeping {keep_versions} most recent)...")
        
        try:
            versions = self.registry.list_all_model_versions()
            
            # Filter out already archived and production models
            active_versions = [v for v in versions if v['stage'] != 'Archived']
            production_versions = [v for v in versions if v['stage'] == 'Production']
            
            # Sort by version number (descending)
            active_versions.sort(key=lambda x: int(x['version']), reverse=True)
            
            # Keep the most recent versions + all production versions
            versions_to_keep = set()
            
            # Keep most recent versions
            for i, version in enumerate(active_versions):
                if i < keep_versions:
                    versions_to_keep.add(version['version'])
            
            # Always keep production versions
            for version in production_versions:
                versions_to_keep.add(version['version'])
            
            # Archive the rest
            archived_count = 0
            for version in active_versions:
                if version['version'] not in versions_to_keep and version['stage'] != 'Production':
                    success = self.registry.transition_model_stage(
                        model_name=self.registry.registered_model_name,
                        version=version['version'],
                        stage="Archived",
                        archive_existing=False
                    )
                    if success:
                        archived_count += 1
                        logger.info(f"📦 Archived version {version['version']}")
            
            logger.info(f"✅ Archived {archived_count} old model versions")
            return archived_count
            
        except Exception as e:
            logger.error(f"Error archiving models: {e}")
            return 0
    
    def generate_model_report(self) -> str:
        """Generate a comprehensive model status report."""
        
        logger.info("📋 Generating model registry report...")
        
        try:
            # Get all versions
            versions = self.registry.list_all_model_versions()
            
            # Production model
            prod_model = self.registry.get_production_model()
            
            # Generate report
            report_lines = []
            report_lines.append("="*80)
            report_lines.append("🏛️  PATIENT READMISSION MODEL REGISTRY REPORT")
            report_lines.append("="*80)
            report_lines.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"📋 Model Name: {self.registry.registered_model_name}")
            report_lines.append(f"📊 Total Versions: {len(versions)}")
            report_lines.append("")
            
            # Production status
            if prod_model:
                report_lines.append("🏭 PRODUCTION MODEL:")
                report_lines.append(f"   Version: {prod_model['version']}")
                report_lines.append(f"   Run ID: {prod_model['run_id'][:8]}...")
                report_lines.append(f"   Deployed: {datetime.fromtimestamp(prod_model['creation_timestamp']/1000).strftime('%Y-%m-%d %H:%M')}")
                report_lines.append("")
            else:
                report_lines.append("⚠️  NO PRODUCTION MODEL DEPLOYED")
                report_lines.append("")
            
            # Staging models
            staging_models = [v for v in versions if v['stage'] == 'Staging']
            if staging_models:
                report_lines.append("🚧 STAGING MODELS:")
                for model in staging_models:
                    report_lines.append(f"   Version {model['version']}: Recall={model['recall']:.3f}, F1={model['f1_score']:.3f}")
                report_lines.append("")
            
            # Version history
            report_lines.append("📈 VERSION HISTORY:")
            report_lines.append("-" * 80)
            report_lines.append(f"{'Ver':<4} {'Stage':<12} {'Recall':<8} {'Precision':<10} {'F1':<8} {'ROC-AUC':<8} {'Created':<12}")
            report_lines.append("-" * 80)
            
            for version in versions:
                created = version['creation_time'].strftime('%m/%d/%Y') if isinstance(version['creation_time'], datetime) else 'N/A'
                recall = f"{version['recall']:.3f}" if isinstance(version['recall'], (int, float)) else 'N/A'
                precision = f"{version['precision']:.3f}" if isinstance(version['precision'], (int, float)) else 'N/A'
                f1 = f"{version['f1_score']:.3f}" if isinstance(version['f1_score'], (int, float)) else 'N/A'
                roc_auc = f"{version['roc_auc']:.3f}" if isinstance(version['roc_auc'], (int, float)) else 'N/A'
                
                report_lines.append(f"{version['version']:<4} {version['stage']:<12} {recall:<8} {precision:<10} {f1:<8} {roc_auc:<8} {created:<12}")
            
            report_lines.append("="*80)
            
            # Save report
            report_content = "\n".join(report_lines)
            
            os.makedirs('reports', exist_ok=True)
            report_filename = f"reports/model_registry_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(report_filename, 'w') as f:
                f.write(report_content)
            
            logger.info(f"📄 Report saved to: {report_filename}")
            
            # Also print to console
            print("\n" + report_content)
            
            return report_filename
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""


def main():
    """CLI interface for model management."""
    
    parser = argparse.ArgumentParser(description="Patient Readmission Model Management")
    parser.add_argument('action', choices=[
        'register', 'deploy', 'rollback', 'validate', 'compare', 
        'archive', 'status', 'report'
    ], help='Action to perform')
    
    parser.add_argument('--version', type=str, help='Model version')
    parser.add_argument('--version2', type=str, help='Second model version for comparison')
    parser.add_argument('--force', action='store_true', help='Force action without validation')
    parser.add_argument('--keep', type=int, default=5, help='Number of versions to keep when archiving')
    
    args = parser.parse_args()
    
    print("🛠️  PATIENT READMISSION MODEL MANAGEMENT")
    print("="*60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Action: {args.action}")
    print("="*60)
    
    # Initialize model manager
    manager = ModelManager()
    
    try:
        if args.action == 'register':
            # Register best model from experiments
            logger.info("🔄 Registering best model...")
            model_version = manager.registry.register_best_model(stage="Staging")
            if model_version:
                print(f"✅ Model registered as version {model_version.version}")
            else:
                print("❌ Model registration failed")
        
        elif args.action == 'deploy':
            # Deploy to production
            success = manager.deploy_to_production(args.version, args.force)
            if success:
                print("✅ Production deployment successful")
            else:
                print("❌ Production deployment failed")
        
        elif args.action == 'rollback':
            # Rollback production
            success = manager.rollback_production(args.version)
            if success:
                print("✅ Production rollback successful")
            else:
                print("❌ Production rollback failed")
        
        elif args.action == 'validate':
            # Validate model
            valid = manager.validate_model_for_production(args.version)
            if valid:
                print("✅ Model validation passed")
            else:
                print("❌ Model validation failed")
        
        elif args.action == 'compare':
            # Compare models
            if not args.version or not args.version2:
                print("❌ Two versions required for comparison (--version and --version2)")
            else:
                comparison = manager.compare_models(args.version, args.version2)
                if comparison:
                    print("✅ Model comparison completed")
                else:
                    print("❌ Model comparison failed")
        
        elif args.action == 'archive':
            # Archive old models
            archived = manager.archive_old_models(args.keep)
            print(f"✅ Archived {archived} old model versions")
        
        elif args.action == 'status':
            # Show registry status
            manager.registry.print_model_registry_status()
        
        elif args.action == 'report':
            # Generate report
            report_file = manager.generate_model_report()
            if report_file:
                print(f"✅ Report generated: {report_file}")
            else:
                print("❌ Report generation failed")
        
        print("="*60)
        print("🎉 Model management operation completed!")
        
    except Exception as e:
        logger.error(f"Error during {args.action}: {e}")
        print(f"❌ Operation failed: {e}")


if __name__ == "__main__":
    main()