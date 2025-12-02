"""
YOLOv8 Fine-tuning Script for Chart Pattern Detection

Trains YOLOv8 model on chart pattern dataset for improved pattern recognition.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChartPatternTrainer:
    """Trainer for YOLOv8 chart pattern detection model"""
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        dataset_path: Optional[str] = None,
        output_dir: str = "/app/models"
    ):
        """
        Initialize chart pattern trainer
        
        Args:
            model_name: Base YOLOv8 model to fine-tune
            dataset_path: Path to dataset directory (YOLO format)
            output_dir: Directory to save trained model
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
    
    def prepare_dataset_structure(self) -> Dict[str, Any]:
        """
        Prepare dataset structure for YOLOv8 training
        
        Expected structure:
        dataset/
          train/
            images/
            labels/
          val/
            images/
            labels/
          test/
            images/
            labels/
          data.yaml
        
        Returns:
            Dataset configuration
        """
        if not self.dataset_path:
            raise ValueError("Dataset path not specified")
        
        dataset_path = Path(self.dataset_path)
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Check for data.yaml
        data_yaml = dataset_path / "data.yaml"
        if not data_yaml.exists():
            self.logger.warning("data.yaml not found, creating default configuration")
            self._create_data_yaml(data_yaml)
        
        return {
            "path": str(dataset_path),
            "yaml": str(data_yaml)
        }
    
    def _create_data_yaml(self, yaml_path: Path):
        """Create default data.yaml for chart patterns"""
        import yaml
        
        config = {
            "path": str(yaml_path.parent),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 13,  # Number of classes
            "names": [
                "head_and_shoulders",
                "double_top",
                "double_bottom",
                "triangle_ascending",
                "triangle_descending",
                "triangle_symmetrical",
                "flag_bullish",
                "flag_bearish",
                "pennant",
                "wedge_rising",
                "wedge_falling",
                "cup_and_handle",
                "inverse_cup_and_handle"
            ]
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.logger.info(f"Created data.yaml at {yaml_path}")
    
    def train(
        self,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        device: Optional[str] = None,
        patience: int = 50,
        save_period: int = 10
    ) -> Path:
        """
        Train YOLOv8 model on chart pattern dataset
        
        Args:
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            device: Device to use ('cpu', 'cuda', '0', '1', etc.)
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
        
        Returns:
            Path to trained model
        """
        try:
            # Prepare dataset
            dataset_config = self.prepare_dataset_structure()
            
            # Load base model
            self.logger.info(f"Loading base model: {self.model_name}")
            model = YOLO(self.model_name)
            
            # Determine device
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.logger.info(f"Training on device: {device}")
            
            # Train model
            self.logger.info("Starting training...")
            results = model.train(
                data=dataset_config["yaml"],
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                patience=patience,
                save_period=save_period,
                project=str(self.output_dir),
                name="chart_patterns",
                exist_ok=True
            )
            
            # Get best model path
            best_model_path = Path(results.save_dir) / "weights" / "best.pt"
            
            # Copy to output directory
            output_model_path = self.output_dir / "chart_patterns.pt"
            import shutil
            shutil.copy(best_model_path, output_model_path)
            
            self.logger.info(f"✅ Training complete! Model saved to: {output_model_path}")
            
            return output_model_path
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise
    
    def validate(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate trained model
        
        Args:
            model_path: Path to model (default: latest trained model)
        
        Returns:
            Validation metrics
        """
        if model_path is None:
            model_path = self.output_dir / "chart_patterns.pt"
        
        if not Path(model_path).exists():
            raise ValueError(f"Model not found: {model_path}")
        
        # Load model
        model = YOLO(str(model_path))
        
        # Prepare dataset
        dataset_config = self.prepare_dataset_structure()
        
        # Validate
        results = model.val(data=dataset_config["yaml"])
        
        return {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr
        }


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 for chart pattern detection")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory (YOLO format)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base YOLOv8 model (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cpu, cuda, 0, 1, etc.)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/app/models",
        help="Output directory for trained model"
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ChartPatternTrainer(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output
    )
    
    # Train
    model_path = trainer.train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )
    
    # Validate
    print("\n" + "=" * 60)
    print("Validating trained model...")
    print("=" * 60)
    
    metrics = trainer.validate(str(model_path))
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics['mAP50']:.4f}")
    print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    
    print(f"\n✅ Model ready: {model_path}")


if __name__ == "__main__":
    main()
