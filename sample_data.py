"""
Sample Data Generator for OMR Evaluation System
Creates sample students, exams, and answer keys for testing
"""
import json
import random
from datetime import datetime, timedelta
from database import SessionLocal, init_database
from models import Student, Exam, SystemConfig
from crud import StudentCRUD, ExamCRUD

class SampleDataGenerator:
    def __init__(self):
        self.db = SessionLocal()
        
    def generate_sample_students(self, count: int = 50):
        """Generate sample students"""
        print(f"🧑‍🎓 Generating {count} sample students...")
        
        first_names = [
            "John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa",
            "William", "Jennifer", "James", "Mary", "Christopher", "Patricia", "Daniel",
            "Linda", "Matthew", "Elizabeth", "Anthony", "Barbara", "Mark", "Susan",
            "Donald", "Jessica", "Steven", "Dorothy", "Paul", "Ashley", "Andrew", "Kimberly"
        ]
        
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
            "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
            "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
            "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"
        ]
        
        students_created = 0
        
        for i in range(count):
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            
            student_data = {
                "student_id": f"STU{i+1:04d}",
                "name": f"{first_name} {last_name}",
                "email": f"{first_name.lower()}.{last_name.lower()}.{i+1:04d}@university.edu",
                "phone": f"+1{random.randint(1000000000, 9999999999)}"
            }
            
            try:
                student = StudentCRUD.create_student(self.db, student_data)
                if student:
                    students_created += 1
            except Exception as e:
                print(f"Error creating student {student_data['student_id']}: {e}")
        
        print(f"✅ Created {students_created} students")
        return students_created
    
    def generate_sample_exams(self):
        """Generate sample exams with different configurations"""
        print("📋 Generating sample exams...")
        
        exam_templates = [
            {
                "exam_name": "Midterm Examination 2024",
                "exam_date": datetime.now() - timedelta(days=30),
                "description": "Comprehensive midterm covering all subjects"
            },
            {
                "exam_name": "Final Examination 2024",
                "exam_date": datetime.now() - timedelta(days=7),
                "description": "Final examination for semester assessment"
            },
            {
                "exam_name": "Practice Test - Set A",
                "exam_date": datetime.now() - timedelta(days=60),
                "description": "Practice test for exam preparation"
            },
            {
                "exam_name": "Mock Examination",
                "exam_date": datetime.now() - timedelta(days=14),
                "description": "Mock exam to simulate real conditions"
            }
        ]
        
        # Subject configurations
        subject_configs = [
            {
                "Mathematics": 20,
                "Physics": 20,
                "Chemistry": 20,
                "Biology": 20,
                "English": 20
            },
            {
                "Computer Science": 25,
                "Data Structures": 25,
                "Algorithms": 25,
                "Database Systems": 25
            },
            {
                "History": 20,
                "Geography": 20,
                "Political Science": 20,
                "Economics": 20,
                "Sociology": 20
            }
        ]
        
        exams_created = 0
        
        for i, template in enumerate(exam_templates):
            subjects = subject_configs[i % len(subject_configs)]
            
            # Generate answer keys for different versions
            answer_keys = {}
            sheet_versions = ["A", "B", "C", "D"]
            
            for version in sheet_versions:
                answer_key = {}
                for q in range(1, 101):
                    # Generate random but consistent answers
                    random.seed(hash(f"{template['exam_name']}-{version}-{q}"))
                    answer_key[f"Q{q}"] = random.choice(["A", "B", "C", "D"])
                answer_keys[version] = answer_key
            
            exam_data = {
                "exam_name": template["exam_name"],
                "exam_date": template["exam_date"],
                "total_questions": 100,
                "subjects": subjects,
                "sheet_versions": sheet_versions,
                "answer_keys": answer_keys
            }
            
            try:
                exam = ExamCRUD.create_exam(self.db, exam_data)
                if exam:
                    exams_created += 1
                    print(f"✅ Created exam: {template['exam_name']}")
            except Exception as e:
                print(f"Error creating exam {template['exam_name']}: {e}")
        
        print(f"✅ Created {exams_created} exams")
        return exams_created
    
    def generate_answer_key_files(self):
        """Generate answer key files for reference"""
        print("🔑 Generating answer key files...")
        
        # Get all exams
        exams = ExamCRUD.get_exams(self.db)
        
        for exam in exams:
            if exam.answer_keys:
                filename = f"answer_keys_{exam.exam_name.replace(' ', '_').lower()}.json"
                
                answer_key_data = {
                    "exam_id": exam.id,
                    "exam_name": exam.exam_name,
                    "exam_date": exam.exam_date.isoformat(),
                    "total_questions": exam.total_questions,
                    "subjects": exam.subjects,
                    "answer_keys": exam.answer_keys,
                    "generated_at": datetime.now().isoformat()
                }
                
                with open(filename, 'w') as f:
                    json.dump(answer_key_data, f, indent=2)
                
                print(f"✅ Generated {filename}")
    
    def create_system_configurations(self):
        """Create additional system configurations"""
        print("⚙️ Creating system configurations...")
        
        additional_configs = [
            {
                "config_key": "processing_timeout_seconds",
                "config_value": "300",
                "description": "Maximum time allowed for processing a single OMR sheet"
            },
            {
                "config_key": "batch_processing_enabled",
                "config_value": "true",
                "description": "Enable batch processing of multiple OMR sheets"
            },
            {
                "config_key": "auto_backup_enabled",
                "config_value": "true",
                "description": "Automatically backup processed results"
            },
            {
                "config_key": "quality_threshold",
                "config_value": "0.8",
                "description": "Minimum quality threshold for image acceptance"
            },
            {
                "config_key": "notification_email",
                "config_value": "admin@university.edu",
                "description": "Email address for system notifications"
            }
        ]
        
        configs_created = 0
        
        for config in additional_configs:
            try:
                existing = self.db.query(SystemConfig).filter(
                    SystemConfig.config_key == config["config_key"]
                ).first()
                
                if not existing:
                    db_config = SystemConfig(**config)
                    self.db.add(db_config)
                    configs_created += 1
            except Exception as e:
                print(f"Error creating config {config['config_key']}: {e}")
        
        self.db.commit()
        print(f"✅ Created {configs_created} system configurations")
        return configs_created
    
    def generate_sample_omr_template(self):
        """Generate a sample OMR template description"""
        print("📄 Generating OMR template description...")
        
        template_info = {
            "template_name": "Standard 100-Question OMR Sheet",
            "version": "1.0",
            "dimensions": {
                "width_mm": 210,
                "height_mm": 297,
                "dpi": 300
            },
            "layout": {
                "header_section": {
                    "student_id_bubbles": {
                        "position": "top_left",
                        "digits": 8,
                        "options_per_digit": 10
                    },
                    "sheet_version_bubbles": {
                        "position": "top_right",
                        "options": ["A", "B", "C", "D"]
                    }
                },
                "question_section": {
                    "total_questions": 100,
                    "options_per_question": 4,
                    "layout_type": "grid",
                    "columns": 4,
                    "questions_per_column": 25
                }
            },
            "subjects": {
                "Subject_1": {"questions": "1-20", "column": 1},
                "Subject_2": {"questions": "21-40", "column": 2},
                "Subject_3": {"questions": "41-60", "column": 3},
                "Subject_4": {"questions": "61-80", "column": 4},
                "Subject_5": {"questions": "81-100", "column": 1}
            },
            "bubble_specifications": {
                "diameter_mm": 4,
                "spacing_mm": 8,
                "fill_threshold": 0.6,
                "detection_method": "contour_analysis"
            },
            "quality_requirements": {
                "min_resolution": "600x800",
                "max_skew_degrees": 5,
                "min_contrast": 0.3,
                "supported_formats": ["JPG", "PNG", "PDF"]
            }
        }
        
        with open("omr_template_specification.json", 'w') as f:
            json.dump(template_info, f, indent=2)
        
        print("✅ Generated omr_template_specification.json")
    
    def close(self):
        """Close database connection"""
        self.db.close()

def main():
    """Main function to generate all sample data"""
    print("🚀 Starting Sample Data Generation")
    print("=" * 50)
    
    # Initialize database
    init_database()
    
    # Create generator
    generator = SampleDataGenerator()
    
    try:
        # Generate sample data
        generator.generate_sample_students(50)
        generator.generate_sample_exams()
        generator.generate_answer_key_files()
        generator.create_system_configurations()
        generator.generate_sample_omr_template()
        
        print("\n🎉 Sample data generation completed successfully!")
        print("=" * 50)
        print("Generated files:")
        print("- Student records in database")
        print("- Exam records in database")
        print("- Answer key JSON files")
        print("- System configurations")
        print("- OMR template specification")
        
    except Exception as e:
        print(f"❌ Error during sample data generation: {e}")
    finally:
        generator.close()

if __name__ == "__main__":
    main()