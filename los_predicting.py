"""
Hospital Length of Stay (LOS) Prediction Pipeline
Complete end-to-end prediction system
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, TweedieRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from PipeLine import run_training_pipeline_single

warnings.filterwarnings('ignore')

# Try to import XGBoost, but don't fail if it's not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    print(f"⚠️  XGBoost not available: {e}")
    print("   Continuing with other models (Poisson, Negative Binomial, Random Forest, Gradient Boosting)\n")
    XGBOOST_AVAILABLE = False

# Set random seed for reproducibility
np.random.seed(42)

# Optional styling for any plots you DO open
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


class LOSPredictionPipeline:
    """
    End-to-end pipeline for predicting Hospital Length of Stay (LOS)
    """

    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def load_and_prepare_data(self, raw_df):
        """Load cleaned data from pipeline and prepare for modeling"""
        print("=" * 70)
        print("LOADING AND PREPARING DATA")
        print("=" * 70)
        print("\nCalling run_training_pipeline_single()...")

        df_clean = run_training_pipeline_single(raw_df)

        print(f"\n✓ Data loaded successfully!")
        print(f"  Shape: {df_clean.shape}")
        print(f"  Columns: {df_clean.shape[1]}")
        print(f"  Rows: {df_clean.shape[0]}")

        # Check if target exists
        if 'spell_episode_los' not in df_clean.columns:
            raise ValueError("Target column 'spell_episode_los' not found!")

        print(f"\nTarget variable (spell_episode_los) statistics:")
        print(df_clean['spell_episode_los'].describe())

        # Separate features and target
        y = df_clean['spell_episode_los']
        X = df_clean.drop('spell_episode_los', axis=1)

        # Handle any remaining categorical variables
        print(f"\n✓ Encoding categorical variables...")
        X = pd.get_dummies(X, drop_first=True)
        self.feature_names = X.columns.tolist()

        print(f"  Final feature count: {len(self.feature_names)}")

        return X, y

    def split_and_scale_data(self, X, y, test_size=0.2):
        """Split data into train/test and scale features"""
        print("\n" + "=" * 70)
        print("SPLITTING AND SCALING DATA")
        print("=" * 70)
        print(f"\nSplit ratio: {100 * (1 - test_size):.0f}% train / {100 * test_size:.0f}% test")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features (important for GLMs)
        print(f"\n✓ Scaling features using StandardScaler...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"\nDataset sizes:")
        print(f"  Train: {len(self.X_train):,} samples")
        print(f"  Test:  {len(self.X_test):,} samples")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def build_models(self):
        """Initialize all prediction models"""
        print("\n" + "=" * 70)
        print("BUILDING MODELS")
        print("=" * 70)

        # 1. Poisson Regression (for count data)
        print("\n1. Poisson Regression")
        print("   - Designed for count data")
        print("   - Assumes constant rate of events")
        self.models['Poisson'] = PoissonRegressor(
            max_iter=500,  # Reduced from 1000
            alpha=0.1
        )

        # 2. Negative Binomial (Tweedie approximation)
        print("\n2. Negative Binomial Regression")
        print("   - Handles overdispersed count data")
        print("   - More flexible than Poisson")
        self.models['Negative_Binomial'] = TweedieRegressor(
            power=1.5,
            alpha=0.1,
            max_iter=500  # Reduced from 1000
        )

        # 3. Random Forest (reduced trees for speed)
        print("\n3. Random Forest")
        print("   - Ensemble of 100 decision trees")
        print("   - Captures non-linear relationships")
        self.models['Random_Forest'] = RandomForestRegressor(
            n_estimators=100,  # Reduced from 200
            max_depth=12,      # Reduced from 15
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        # 4. Gradient Boosting (faster settings)
        print("\n4. Gradient Boosting")
        print("   - Sequential tree building")
        print("   - Corrects previous errors")
        self.models['Gradient_Boosting'] = GradientBoostingRegressor(
            n_estimators=100,  # Reduced from 200
            learning_rate=0.1,
            max_depth=4,       # Reduced from 5
            min_samples_split=10,
            random_state=42,
            verbose=0
        )

        # 5. XGBoost (only if available, with faster settings)
        if XGBOOST_AVAILABLE:
            print("\n5. XGBoost")
            print("   - Optimized gradient boosting")
            print("   - Regularization + parallel processing")
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,  # Reduced from 200
                learning_rate=0.1,
                max_depth=4,       # Reduced from 5
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        else:
            print("\n5. XGBoost - SKIPPED (not available)")

        print(f"\n✓ Initialized {len(self.models)} models successfully!")

    def train_and_evaluate(self):
        """Train all models and evaluate performance"""
        print("\n" + "=" * 70)
        print("TRAINING AND EVALUATING MODELS")
        print("=" * 70)

        for i, (name, model) in enumerate(self.models.items(), 1):
            print(f"\n[{i}/{len(self.models)}] {name}")
            print("-" * 50)

            # Use scaled data for GLMs, original for tree-based
            if name in ['Poisson', 'Negative_Binomial']:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test

            # Train
            print(f"Training...")
            model.fit(X_train_use, self.y_train)

            # Predict
            y_pred_train = model.predict(X_train_use)
            y_pred_test = model.predict(X_test_use)

            # Ensure predictions are non-negative
            y_pred_train = np.maximum(y_pred_train, 0)
            y_pred_test = np.maximum(y_pred_test, 0)

            # Evaluate
            train_metrics = self._calculate_metrics(self.y_train, y_pred_train)
            test_metrics = self._calculate_metrics(self.y_test, y_pred_test)

            # Store results
            self.results[name] = {
                'model': model,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }

            # Print results
            print("\nResults:")
            print(
                f"  Train - MAE: {train_metrics['MAE']:.4f} | "
                f"RMSE: {train_metrics['RMSE']:.4f} | R²: {train_metrics['R2']:.4f}"
            )
            print(
                f"  Test  - MAE: {test_metrics['MAE']:.4f} | "
                f"RMSE: {test_metrics['RMSE']:.4f} | R²: {test_metrics['R2']:.4f}"
            )

            # Check for overfitting
            mae_diff = train_metrics['MAE'] - test_metrics['MAE']
            if mae_diff < -0.5:
                print(f"  ⚠️  Possible underfitting (test better than train)")
            elif mae_diff > 1.0:
                print(f"  ⚠️  Possible overfitting (train much better than test)")
            else:
                print(f"  ✓ Good generalization")

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }

    def predict_with_details(self, model_name, X_new):
        """
        Make predictions with detailed output (integer days and decimal days)

        Parameters:
        -----------
        model_name : str
            Name of the model to use for prediction
        X_new : pd.DataFrame or np.ndarray
            New data to predict on (raw features, like training before dummies)

        Returns:
        --------
        pd.DataFrame with columns: predicted_days_decimal, predicted_days_rounded
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.results.keys())}")

        model = self.results[model_name]['model']

        # If DataFrame, apply same dummy logic + align columns
        if isinstance(X_new, pd.DataFrame):
            X_new = pd.get_dummies(X_new, drop_first=True)
            X_new = X_new.reindex(columns=self.feature_names, fill_value=0)

        # Use appropriate data format
        if model_name in ['Poisson', 'Negative_Binomial']:
            X_scaled = self.scaler.transform(X_new)
            predictions_decimal = model.predict(X_scaled)
        else:
            predictions_decimal = model.predict(X_new)

        # Ensure non-negative
        predictions_decimal = np.maximum(predictions_decimal, 0)

        # Create results dataframe
        results_df = pd.DataFrame({
            'predicted_days_decimal': predictions_decimal,
            'predicted_days_rounded': np.round(predictions_decimal).astype(int)
        })

        return results_df

    def cross_validate_best_model(self, cv_folds=3):
        """Perform k-fold cross-validation on best performing model"""
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION")
        print("=" * 70)

        # Find best model based on test MAE
        best_model_name = min(
            self.results.items(),
            key=lambda x: x[1]['test_metrics']['MAE']
        )[0]

        print(f"\n✓ Best model (lowest test MAE): {best_model_name}")
        print(f"  Performing {cv_folds}-fold cross-validation...")

        best_model = self.models[best_model_name]

        # Use appropriate data
        if best_model_name in ['Poisson', 'Negative_Binomial']:
            X_use = np.vstack([self.X_train_scaled, self.X_test_scaled])
        else:
            X_use = pd.concat([self.X_train, self.X_test])

        y_use = pd.concat([self.y_train, self.y_test])

        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Calculate CV scores
        print(f"\nCalculating cross-validation scores...")
        cv_mae = -cross_val_score(best_model, X_use, y_use,
                                  cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_rmse = np.sqrt(-cross_val_score(best_model, X_use, y_use,
                                           cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1))
        cv_r2 = cross_val_score(best_model, X_use, y_use,
                                cv=kfold, scoring='r2', n_jobs=-1)

        print(f"\n{cv_folds}-Fold Cross-Validation Results:")
        print(f"  MAE:  {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
        print(f"  RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
        print(f"  R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

        return cv_mae, cv_rmse, cv_r2

    def get_feature_importance(self, top_n=20):
        """Get feature importance for tree-based models (non-blocking plot)"""
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 70)

        # Find best tree-based model
        tree_models = ['Random_Forest', 'Gradient_Boosting', 'XGBoost']
        available_tree_models = [m for m in tree_models if m in self.results]

        if not available_tree_models:
            print("\n⚠️  No tree-based models available for feature importance")
            return None

        # Get best tree model by test MAE
        best_tree_model = min(
            [(m, self.results[m]['test_metrics']['MAE']) for m in available_tree_models],
            key=lambda x: x[1]
        )[0]

        print(f"\n✓ Using {best_tree_model} for feature importance")

        model = self.results[best_tree_model]['model']

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            # Create dataframe
            feature_imp_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(top_n)

            print(f"\nTop {top_n} Most Important Features:")
            print("-" * 50)
            for _, row in feature_imp_df.iterrows():
                print(f"  {row['Feature'][:40]:40s} {row['Importance']:.4f}")

            # Plot (save & close, no blocking window)
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(feature_imp_df)), feature_imp_df['Importance'])
            plt.yticks(range(len(feature_imp_df)), feature_imp_df['Feature'])
            plt.xlabel('Importance Score')
            plt.title(f'{best_tree_model} - Top {top_n} Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            filename = f'{best_tree_model.lower()}_feature_importance.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✓ Saved: '{filename}'")

            return feature_imp_df

        return None

    def plot_results(self):
        """Visualize model performance (non-blocking plot)"""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Model Comparison - Test MAE
        ax1 = axes[0, 0]
        model_names = list(self.results.keys())
        test_maes = [self.results[m]['test_metrics']['MAE'] for m in model_names]

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))
        bars1 = ax1.barh(model_names, test_maes, color=colors)
        ax1.set_xlabel('Mean Absolute Error (days)', fontsize=11)
        ax1.set_title('Model Comparison - Test MAE (Lower is Better)', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()

        # Add value labels
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=9)

        # 2. Model Comparison - Test R²
        ax2 = axes[0, 1]
        test_r2s = [self.results[m]['test_metrics']['R2'] for m in model_names]

        bars2 = ax2.barh(model_names, test_r2s, color=colors)
        ax2.set_xlabel('R² Score', fontsize=11)
        ax2.set_title('Model Comparison - Test R² (Higher is Better)', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()

        # Add value labels
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=9)

        # 3. Actual vs Predicted (Best Model)
        ax3 = axes[1, 0]
        best_model_name = min(
            self.results.items(),
            key=lambda x: x[1]['test_metrics']['MAE']
        )[0]

        y_pred_best = self.results[best_model_name]['y_pred_test']

        ax3.scatter(self.y_test, y_pred_best, alpha=0.3, s=10, color='steelblue')
        max_val = max(self.y_test.max(), y_pred_best.max())
        ax3.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax3.set_xlabel('Actual LOS (days)', fontsize=11)
        ax3.set_ylabel('Predicted LOS (days)', fontsize=11)
        ax3.set_title(f'Actual vs Predicted - {best_model_name}', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)

        # 4. Residuals (Best Model)
        ax4 = axes[1, 1]
        residuals = self.y_test - y_pred_best

        ax4.scatter(y_pred_best, residuals, alpha=0.3, s=10, color='coral')
        ax4.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Residual')
        ax4.set_xlabel('Predicted LOS (days)', fontsize=11)
        ax4.set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
        ax4.set_title(f'Residual Plot - {best_model_name}', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('los_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 'los_prediction_results.png'")

    def print_summary(self, experiment_name="Outliers"):
        """
        Print comprehensive summary of results and
        save comparison table into model_comparison_results.csv
        """
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        # Build comparison table (same structure as your screenshot)
        comparison_rows = []
        for model_name in self.results:
            comparison_rows.append({
                'MAE':  self.results[model_name]['test_metrics']['MAE'],
                'RMSE': self.results[model_name]['test_metrics']['RMSE'],
                'R2':   self.results[model_name]['test_metrics']['R2'],
                'Model': model_name,
                'Experiment': experiment_name
            })

        comparison_df = pd.DataFrame(comparison_rows).sort_values('MAE').reset_index(drop=True)

        # Save to CSV (this is your "compare" file)
        comparison_df.to_csv('model_comparison_results.csv', index=True)
        print("\n✓ Saved comparison table to 'model_comparison_results.csv'")

        print("\nModel Comparison Results:")
        print("-" * 70)
        try:
            print(comparison_df.to_markdown(index=True, floatfmt=".6f"))
        except Exception:
            print(comparison_df.to_string(index=True, float_format=lambda x: f"{x:.6f}"))
        print("-" * 70)

        # Best model
        best_model = comparison_df.iloc[0]['Model']
        mae = comparison_df.iloc[0]['MAE']

        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   RMSE: {comparison_df.iloc[0]['RMSE']:.4f}")
        print(f"   R²:   {comparison_df.iloc[0]['R2']:.4f}")

        # Sample predictions from test set
        print(f"\n📋 SAMPLE PREDICTIONS (first 10 test cases):")
        print("-" * 70)

        best_y_pred = self.results[best_model]['y_pred_test']
        y_test_array = self.y_test.values

        sample_rows = []
        for i in range(min(10, len(y_test_array))):
            sample_rows.append({
                "Actual (days)": y_test_array[i],
                "Predicted (decimal)": best_y_pred[i],
                "Predicted (rounded)": int(round(best_y_pred[i]))
            })

        sample_df = pd.DataFrame(sample_rows)
        try:
            print(sample_df.to_markdown(index=False, floatfmt=".3f"))
        except Exception:
            print(sample_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("-" * 70)

        # Interpretation (optional, quick)
        print("\n📊 INTERPRETATION:")
        print(f"   - On average, predictions are off by {mae:.2f} days")

        return comparison_df


def main(raw_df):
    """
    Run the complete prediction pipeline

    Parameters:
    -----------
    raw_df : pd.DataFrame
        Raw hospital data with 'spell_episode_los' target column

    Returns:
    --------
    pipeline : LOSPredictionPipeline
        Trained pipeline object with all results
    """

    print("\n" + "=" * 70)
    print("HOSPITAL LENGTH OF STAY PREDICTION PIPELINE")
    print("=" * 70)

    # Initialize pipeline
    pipeline = LOSPredictionPipeline()

    # Load and prepare data
    X, y = pipeline.load_and_prepare_data(raw_df)

    # Split and scale
    pipeline.split_and_scale_data(X, y, test_size=0.2)

    # Build models
    pipeline.build_models()

    # Train and evaluate
    pipeline.train_and_evaluate()

    # Cross-validate best model (reduced to 3-fold for speed)
    pipeline.cross_validate_best_model(cv_folds=3)

    # Get feature importance (optional)
    pipeline.get_feature_importance(top_n=20)

    # Plot results
    pipeline.plot_results()

    # Print summary (saves comparison CSV)
    pipeline.print_summary(experiment_name="Outliers")

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print("\n📁 Files saved:")
    print("  - model_comparison_results.csv (comparison table)")
    print("  - los_prediction_results.png (visualizations)")
    print("  - <tree_model>_feature_importance.png")
    print()

    return pipeline


if __name__ == "__main__":
    # Load your raw data
    data_path = os.path.join(os.getcwd(), "wwlLancMsc_data", "wwlLancMsc_data.csv")

    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        raw_data = pd.read_csv(data_path)

        # Run the complete pipeline
        pipeline = main(raw_data)
    else:
        print(f"⚠️  Data file not found at: {data_path}")
        print("\nTo run with your data:")
        print("  raw_data = pd.read_csv('your_data.csv')")
        print("  pipeline = main(raw_data)")
