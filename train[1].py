#!/usr/bin/env python3
"""
Universal AutoML Training Pipeline
Usage: python train.py --data <csv_file> --target <target_column>
"""

import argparse
import pandas as pd
import numpy as np
import json
import yaml
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, 'src')

try:
    from universal_core.data_detector import DataDetector
    from universal_core.problem_detector import ProblemDetector
    from universal_core.smart_preprocessor import SmartPreprocessor
    from automl_engine.trainer import AutoMLTrainer
    from automl_engine.simple_explainer import SimpleExplainer
    from risk_optimizer.cost_calculator import BusinessCostCalculator
    from risk_optimizer.profit_optimizer import ProfitOptimizer
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all modules are created in src/ directory")
    sys.exit(1)

def load_config():
    """Load configuration from YAML"""
    config_path = 'config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {
        'business': {
            'false_negative_cost': 500,
            'false_positive_cost': 50,
            'intervention_cost': 50,
            'customer_value': 500
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Universal AutoML Training')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--problem_type', type=str,
                       choices=['auto', 'binary', 'multiclass', 'regression'],
                       default='auto', help='Problem type (auto-detected if not specified)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--n_trials', type=int, default=30, help='Number of optimization trials')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("🚀 RISKGUARD AUTOML - UNIVERSAL TRAINING PIPELINE")
    print("=" * 70)
    print(f"📁 Dataset: {args.data}")
    print(f"🎯 Target: {args.target}")

    try:
        # 1. LOAD DATA
        print("\n1️⃣ LOADING DATA...")
        df = pd.read_csv(args.data)
        print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

        # 2. DETECT DATA TYPES
        print("\n2️⃣ ANALYZING DATA...")
        detector = DataDetector()
        data_info = detector.detect_column_types(df, args.target)
        print(f"   Numeric features: {len(data_info['numeric_features'])}")
        print(f"   Categorical features: {len(data_info['categorical_features'])}")
        print(f"   Missing data: {'Yes' if data_info['has_missing'] else 'No'}")

        # 3. DETECT PROBLEM TYPE
        print("\n3️⃣ DETECTING PROBLEM TYPE...")
        problem_detector = ProblemDetector()

        if args.problem_type == 'auto':
            problem_type = problem_detector.detect_problem_type(df[args.target])
        else:
            problem_type = f"{args.problem_type}_classification" if args.problem_type in ['binary', 'multiclass'] else args.problem_type

        metrics = problem_detector.get_problem_metrics(problem_type)
        print(f"   Problem type: {problem_type}")
        print(f"   Primary metric: {metrics.get('primary', 'N/A')}")

        # 4. PREPARE DATA
        print("\n4️⃣ PREPROCESSING DATA...")
        X = df.drop(columns=[args.target])
        y = df[args.target]

        # Handle categorical encoding for y if needed
        if 'classification' in problem_type and y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=args.target)
            print(f"   Encoded target variable")

        preprocessor = SmartPreprocessor()
        X_processed = preprocessor.fit_transform(
            X,
            data_info['numeric_features'],
            data_info['categorical_features']
        )
        print(f"   Processed features: {X_processed.shape[1]}")

        # 5. SPLIT DATA
        print("\n5️⃣ SPLITTING DATA...")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=args.test_size, random_state=42,
            stratify=y if 'classification' in problem_type else None
        )
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")

        # 6. TRAIN MODEL WITH AUTOML
        print(f"\n6️⃣ TRAINING MODEL ({args.n_trials} trials)...")
        trainer = AutoMLTrainer(problem_type=problem_type, n_trials=args.n_trials)
        best_model, best_score, best_params = trainer.optimize(X_train, y_train)

        print(f"   Best model: {type(best_model).__name__}")
        print(f"   Best CV score: {best_score:.4f}")

        # 7. EVALUATE ON TEST SET
        print("\n7️⃣ EVALUATING MODEL...")
        from sklearn.metrics import get_scorer

        scorer_name = metrics.get('scoring', 'roc_auc' if problem_type == 'binary_classification' else 'accuracy')
        scorer = get_scorer(scorer_name)
        test_score = scorer(best_model, X_test, y_test)

        print(f"   Test score: {test_score:.4f}")

        # 8. GENERATE EXPLANATIONS
        print("\n8️⃣ GENERATING EXPLANATIONS...")
        explainer = SimpleExplainer(best_model, preprocessor.feature_names)
        explanation = explainer.explain_prediction(X_test, instance_idx=0)

        print(f"   Sample prediction: {explanation['prediction']}")
        if explanation['prediction_proba']:
            print(f"   Prediction confidence: {explanation['prediction_proba']:.2%}")
        
        print(f"   Top 3 important features:")
        for feat in explanation['top_features'][:3]:
            print(f"     • {feat['feature']}: importance={feat['importance']:.4f}")

        # 9. BUSINESS OPTIMIZATION (only for binary classification)
        business_report = None
        if problem_type == 'binary_classification':
            print("\n9️⃣ OPTIMIZING FOR BUSINESS PROFIT...")

            # Get predicted probabilities
            y_prob = best_model.predict_proba(X_test)[:, 1]

            # Load business costs
            config = load_config()
            business_config = config.get('business', {})

            cost_calculator = BusinessCostCalculator(
                false_negative_cost=business_config.get('false_negative_cost', 500),
                false_positive_cost=business_config.get('false_positive_cost', 50),
                intervention_cost=business_config.get('intervention_cost', 50),
                customer_value=business_config.get('customer_value', 500)
            )

            profit_optimizer = ProfitOptimizer(cost_calculator)
            business_report = profit_optimizer.generate_business_report(
                y_test, y_prob, customer_count=len(y_test)
            )

            print(f"   Optimal threshold: {business_report['summary']['optimal_threshold']:.3f}")
            print(f"   Expected profit: ${business_report['summary']['expected_profit']:,.2f}")
            print(f"   ROI: {business_report['summary']['roi']:.1f}%")

        # 10. SAVE RESULTS
        print(f"\n🔟 SAVING RESULTS to {args.output_dir}/...")

        # Save model
        import joblib
        model_path = os.path.join(args.output_dir, 'model.pkl')
        trainer.save_model(model_path)

        # Save metadata with JSON serialization fix
        metadata = {
            'dataset': args.data,
            'target_column': args.target,
            'problem_type': problem_type,
            'training_date': datetime.now().isoformat(),
            'model_type': type(best_model).__name__,
            'cv_score': float(best_score),
            'test_score': float(test_score),
            'best_params': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                          for k, v in best_params.items()} if best_params else {},
            'data_info': {
                'numeric_features': data_info['numeric_features'],
                'categorical_features': data_info['categorical_features'],
                'datetime_features': data_info['datetime_features'],
                'text_features': data_info['text_features'],
                'has_missing': bool(data_info['has_missing']),  # Convert to bool
                'missing_percentage': {k: float(v) for k, v in data_info['missing_percentage'].items()}
            },
            'feature_names': preprocessor.feature_names,
            'sample_explanation': {
                'prediction': float(explanation['prediction']) if explanation['prediction'] is not None else None,
                'prediction_proba': float(explanation['prediction_proba']) if explanation['prediction_proba'] is not None else None,
                'top_features': [
                    {
                        'feature': feat['feature'],
                        'importance': float(feat['importance']),
                        'value': float(feat['value']) if feat['value'] is not None else None,
                        'contribution': float(feat['contribution']) if feat['contribution'] is not None else None
                    }
                    for feat in explanation['top_features']
                ]
            },
            'business_report': business_report
        }

        metadata_path = os.path.join(args.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save feature importance plot
        importance_plot = explainer.plot_importance()
        plot_path = os.path.join(args.output_dir, 'feature_importance.png')
        importance_plot.savefig(plot_path, dpi=150, bbox_inches='tight')

        print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📁 Output directory: {args.output_dir}")
        print(f"📦 Model saved: {model_path}")
        print(f"📊 Metadata: {metadata_path}")
        print(f"📈 Feature importance plot: {plot_path}")
        
        if business_report:
            print(f"\n💰 BUSINESS IMPACT:")
            print(f"   Expected profit: ${business_report['summary']['expected_profit']:,.2f}")
            print(f"   ROI: {business_report['summary']['roi']:.1f}%")
            print(f"\n🎯 RECOMMENDATIONS:")
            for rec in business_report['recommendations']:
                print(f"   • {rec}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
