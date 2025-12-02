"""
Chart Pattern Recognition using YOLOv8

Detects chart patterns (head and shoulders, triangles, flags, etc.) in candlestick charts.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ChartPatternRecognizer:
    """Recognizes chart patterns using YOLOv8 model"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.25
    ):
        """
        Initialize chart pattern recognizer
        
        Args:
            model_path: Path to custom trained YOLOv8 model (optional)
            model_name: YOLOv8 model name (yolov8n.pt, yolov8s.pt, etc.)
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.logger = logger
        
        # Chart pattern classes (can be extended with custom training)
        self.pattern_classes = [
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
    
    def load_model(self):
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO
            
            if self.model is not None:
                return  # Already loaded
            
            if self.model_path and Path(self.model_path).exists():
                self.logger.info(f"Loading custom model from {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                self.logger.info(f"Loading YOLOv8 model: {self.model_name}")
                # Load pre-trained YOLOv8 model
                # For chart patterns, we'll use a base model and can fine-tune later
                self.model = YOLO(self.model_name)
                
                # If we have a custom chart pattern model path, try to load it
                # This would be a fine-tuned model trained on chart patterns
                custom_path = Path("/app/models/chart_patterns.pt")
                if custom_path.exists():
                    self.logger.info(f"Loading fine-tuned chart pattern model from {custom_path}")
                    self.model = YOLO(str(custom_path))
            
            self.logger.info("YOLOv8 model loaded successfully")
            
        except ImportError:
            self.logger.error("ultralytics not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            self.logger.error(f"Error loading YOLOv8 model: {e}", exc_info=True)
            raise
    
    def detect_patterns(
        self,
        image: Image.Image,
        return_image: bool = False
    ) -> Dict[str, Any]:
        """
        Detect chart patterns in image
        
        Args:
            image: PIL Image of candlestick chart
            return_image: Whether to return annotated image
        
        Returns:
            Dict with:
                - patterns: List of detected patterns with confidence
                - count: Number of patterns detected
                - annotated_image: PIL Image (if return_image=True)
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Run inference
            results = self.model(img_array, conf=self.confidence_threshold)
            
            # Parse results
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID and confidence
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Get bounding box coordinates
                        xyxy = box.xyxy[0].cpu().numpy()
                        
                        # Map class ID to pattern name
                        # Note: This assumes custom training with pattern classes
                        # For base YOLOv8, we'd need to map COCO classes or use custom model
                        pattern_name = self._get_pattern_name(cls_id)
                        
                        detections.append({
                            "pattern": pattern_name,
                            "confidence": conf,
                            "bbox": {
                                "x1": float(xyxy[0]),
                                "y1": float(xyxy[1]),
                                "x2": float(xyxy[2]),
                                "y2": float(xyxy[3])
                            }
                        })
            
            result_dict = {
                "patterns": detections,
                "count": len(detections),
                "timestamp": None  # Will be set by caller if needed
            }
            
            # Add annotated image if requested
            if return_image:
                # Plot results on image
                annotated_img = results[0].plot()
                result_dict["annotated_image"] = Image.fromarray(annotated_img)
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}", exc_info=True)
            return {
                "patterns": [],
                "count": 0,
                "error": str(e)
            }
    
    def _get_pattern_name(self, class_id: int) -> str:
        """
        Map class ID to pattern name
        
        Args:
            class_id: YOLOv8 class ID
        
        Returns:
            Pattern name
        """
        # For base YOLOv8, class IDs are COCO classes (not chart patterns)
        # This method should be updated when we have a custom trained model
        # For now, return a generic pattern name
        
        if class_id < len(self.pattern_classes):
            return self.pattern_classes[class_id]
        else:
            return f"pattern_{class_id}"
    
    def analyze_chart(
        self,
        chart_image: Image.Image,
        symbol: str = "UNKNOWN"
    ) -> Dict[str, Any]:
        """
        Analyze chart image and return pattern detection results
        
        Args:
            chart_image: PIL Image of candlestick chart
            symbol: Trading symbol
        
        Returns:
            Analysis results with patterns and recommendations
        """
        detections = self.detect_patterns(chart_image, return_image=True)
        
        # Analyze patterns for trading signals
        bullish_patterns = ["double_bottom", "flag_bullish", "cup_and_handle", "triangle_ascending"]
        bearish_patterns = ["head_and_shoulders", "double_top", "flag_bearish", "triangle_descending"]
        
        bullish_count = sum(1 for p in detections["patterns"] 
                          if p["pattern"] in bullish_patterns)
        bearish_count = sum(1 for p in detections["patterns"] 
                          if p["pattern"] in bearish_patterns)
        
        # Determine overall signal
        signal = "HOLD"
        confidence = 0.5
        
        if bullish_count > bearish_count and bullish_count > 0:
            signal = "BUY"
            confidence = min(0.9, 0.5 + (bullish_count * 0.1))
        elif bearish_count > bullish_count and bearish_count > 0:
            signal = "SELL"
            confidence = min(0.9, 0.5 + (bearish_count * 0.1))
        
        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "patterns_detected": detections["patterns"],
            "pattern_count": detections["count"],
            "bullish_patterns": bullish_count,
            "bearish_patterns": bearish_count,
            "annotated_image": detections.get("annotated_image"),
            "analysis": self._generate_analysis(detections["patterns"])
        }
    
    def _generate_analysis(self, patterns: List[Dict[str, Any]]) -> str:
        """Generate human-readable analysis of detected patterns"""
        if not patterns:
            return "No chart patterns detected."
        
        pattern_names = [p["pattern"] for p in patterns]
        unique_patterns = list(set(pattern_names))
        
        analysis = f"Detected {len(patterns)} pattern(s): {', '.join(unique_patterns)}. "
        
        if len(unique_patterns) > 1:
            analysis += "Multiple patterns suggest complex market structure. "
        
        return analysis
