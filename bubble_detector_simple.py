"""
Simplified bubble detection module for OMR sheets using only OpenCV
Avoids sklearn dependencies that cause DLL loading issues on Windows
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import math


class BubbleDetector:
    """Detects and classifies bubbles in OMR sheets using OpenCV only"""
    
    def __init__(self):
        self.bubble_templates = []
        self.grid_config = {
            "questions_per_subject": 20,
            "subjects": 5,
            "options_per_question": 4,  # A, B, C, D
            "expected_bubbles": 400  # 100 questions * 4 options
        }
        
    def set_grid_config(self, config: Dict[str, Any]):
        """Set OMR grid configuration"""
        self.grid_config.update(config)
        
    def detect_bubbles(self, image: np.ndarray, debug: bool = False) -> List[Dict[str, Any]]:
        """
        Detect bubble locations in the image using OpenCV contour detection
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive threshold to handle varying lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue
                
            # Calculate circularity (4π * area / perimeter²)
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            # Filter based on area and circularity to find bubble-like shapes
            if 100 < area < 2000 and circularity > 0.3:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (should be close to 1 for circles)
                aspect_ratio = w / h if h > 0 else 0
                
                if 0.7 < aspect_ratio < 1.3:  # Nearly square/circular
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    bubbles.append({
                        'center': (center_x, center_y),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'contour': contour,
                        'circularity': circularity
                    })
        
        # Sort bubbles by position (top to bottom, left to right)
        bubbles.sort(key=lambda b: (b['center'][1], b['center'][0]))
        
        if debug:
            print(f"Detected {len(bubbles)} potential bubbles")
            
        return bubbles
    
    def classify_bubbles(self, image: np.ndarray, bubbles: List[Dict[str, Any]], 
                        debug: bool = False) -> List[Dict[str, Any]]:
        """
        Classify bubbles as filled or unfilled using pixel intensity analysis
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        classified_bubbles = []
        
        for bubble in bubbles:
            x, y, w, h = bubble['bbox']
            
            # Extract bubble region with some padding
            padding = 2
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(gray.shape[1], x + w + padding)
            y2 = min(gray.shape[0], y + h + padding)
            
            bubble_region = gray[y1:y2, x1:x2]
            
            if bubble_region.size == 0:
                continue
                
            # Calculate mean intensity in the bubble region
            mean_intensity = np.mean(bubble_region)
            
            # Calculate the percentage of dark pixels (threshold-based)
            _, binary = cv2.threshold(bubble_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dark_pixels = np.sum(binary == 0)
            total_pixels = binary.size
            dark_percentage = dark_pixels / total_pixels if total_pixels > 0 else 0
            
            # Classify based on intensity and dark pixel percentage
            # Lower intensity and higher dark percentage indicate filled bubble
            is_filled = mean_intensity < 180 and dark_percentage > 0.3
            confidence = 1.0 - (mean_intensity / 255.0)  # Simple confidence metric
            
            classified_bubble = bubble.copy()
            classified_bubble.update({
                'is_filled': is_filled,
                'confidence': confidence,
                'mean_intensity': mean_intensity,
                'dark_percentage': dark_percentage
            })
            
            classified_bubbles.append(classified_bubble)
            
        if debug:
            filled_count = sum(1 for b in classified_bubbles if b['is_filled'])
            print(f"Classified {len(classified_bubbles)} bubbles: {filled_count} filled, {len(classified_bubbles) - filled_count} unfilled")
            
        return classified_bubbles
    
    def organize_bubbles_into_grid(self, bubbles: List[Dict[str, Any]], 
                                  debug: bool = False) -> Dict[str, Any]:
        """
        Organize detected bubbles into a grid structure based on position
        """
        if not bubbles:
            return {"questions": [], "grid_info": {}}
            
        # Sort bubbles by position
        sorted_bubbles = sorted(bubbles, key=lambda b: (b['center'][1], b['center'][0]))
        
        # Group bubbles into rows based on Y coordinate
        rows = []
        current_row = []
        row_threshold = 30  # pixels
        
        for bubble in sorted_bubbles:
            if not current_row:
                current_row.append(bubble)
            else:
                # Check if bubble is in the same row
                last_y = current_row[-1]['center'][1]
                current_y = bubble['center'][1]
                
                if abs(current_y - last_y) <= row_threshold:
                    current_row.append(bubble)
                else:
                    # Start new row
                    if current_row:
                        rows.append(sorted(current_row, key=lambda b: b['center'][0]))
                    current_row = [bubble]
        
        # Add the last row
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b['center'][0]))
        
        # Organize into questions (assuming 4 options per question)
        questions = []
        options_per_question = self.grid_config.get("options_per_question", 4)
        
        for row in rows:
            # Group bubbles in this row into questions
            for i in range(0, len(row), options_per_question):
                question_bubbles = row[i:i + options_per_question]
                if len(question_bubbles) == options_per_question:
                    # Find which option is selected (if any)
                    selected_option = None
                    for j, bubble in enumerate(question_bubbles):
                        if bubble.get('is_filled', False):
                            selected_option = chr(ord('A') + j)  # A, B, C, D
                            break
                    
                    questions.append({
                        'question_number': len(questions) + 1,
                        'selected_option': selected_option,
                        'bubbles': question_bubbles,
                        'confidence': min(b.get('confidence', 0) for b in question_bubbles)
                    })
        
        grid_info = {
            'total_questions': len(questions),
            'total_bubbles': len(bubbles),
            'rows_detected': len(rows),
            'expected_questions': self.grid_config.get("questions_per_subject", 20) * self.grid_config.get("subjects", 5)
        }
        
        if debug:
            print(f"Organized {len(bubbles)} bubbles into {len(questions)} questions across {len(rows)} rows")
            
        return {
            "questions": questions,
            "grid_info": grid_info
        }
    
    def process_omr_sheet(self, image: np.ndarray, debug: bool = False) -> Dict[str, Any]:
        """
        Complete OMR processing pipeline
        """
        # Step 1: Detect bubbles
        bubbles = self.detect_bubbles(image, debug=debug)
        
        # Step 2: Classify bubbles
        classified_bubbles = self.classify_bubbles(image, bubbles, debug=debug)
        
        # Step 3: Organize into grid
        result = self.organize_bubbles_into_grid(classified_bubbles, debug=debug)
        
        return result
    
    def map_bubbles_to_answers(self, grid_structure: Dict, classifications: List[bool]) -> Dict[str, Any]:
        """
        Map detected bubbles to student answers
        """
        answers = {}
        flagged_questions = []
        
        if "questions" in grid_structure:
            questions = grid_structure["questions"]
            
            for i, question in enumerate(questions):
                question_num = i + 1
                filled_bubbles = []
                
                # Check which bubbles are filled for this question
                for j, bubble in enumerate(question.get("bubbles", [])):
                    if bubble.get("is_filled", False):
                        # Map to A, B, C, D based on position
                        option = chr(ord('A') + j)
                        filled_bubbles.append(option)
                
                # Determine the answer
                if len(filled_bubbles) == 1:
                    answers[f"Q{question_num}"] = filled_bubbles[0]
                elif len(filled_bubbles) == 0:
                    answers[f"Q{question_num}"] = "BLANK"
                    flagged_questions.append({
                        "question": question_num,
                        "issue": "No answer selected"
                    })
                else:
                    answers[f"Q{question_num}"] = "MULTIPLE"
                    flagged_questions.append({
                        "question": question_num,
                        "issue": f"Multiple answers selected: {', '.join(filled_bubbles)}"
                    })
        
        return {
            "answers": answers,
            "flagged_questions": flagged_questions
        }
    
    def save_debug_image(self, image: np.ndarray, bubbles: List[Dict[str, Any]], 
                        output_path: str):
        """
        Save debug image with detected bubbles highlighted
        """
        debug_image = image.copy()
        if len(debug_image.shape) == 1:
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
            
        for bubble in bubbles:
            x, y, w, h = bubble['bbox']
            center = bubble['center']
            is_filled = bubble.get('is_filled', False)
            
            # Draw bounding box
            color = (0, 255, 0) if is_filled else (0, 0, 255)  # Green for filled, red for unfilled
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            cv2.circle(debug_image, center, 3, color, -1)
            
            # Add confidence text
            confidence = bubble.get('confidence', 0)
            cv2.putText(debug_image, f"{confidence:.2f}", 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imwrite(output_path, debug_image)