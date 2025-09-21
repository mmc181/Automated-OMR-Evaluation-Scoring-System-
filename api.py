"""
FastAPI Backend for OMR Evaluation System
Provides REST API endpoints for file upload, processing, and results management.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import json
import uuid
from datetime import datetime
import shutil

from database import get_db
from models import Student, Exam, ExamResult, AuditLog
from omr_processor import OMRProcessor
from config import Config

# Initialize FastAPI app
app = FastAPI(
    title="OMR Evaluation System API",
    description="Automated OMR sheet evaluation and scoring system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize configuration and OMR processor
config = Config()
omr_processor = OMRProcessor()

# Create upload directories
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.PROCESSED_DIR, exist_ok=True)
os.makedirs(config.EXPORTS_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "OMR Evaluation System API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "upload": "/upload",
            "process": "/process",
            "results": "/results",
            "students": "/students",
            "exams": "/exams"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    exam_id: Optional[str] = Form(None),
    student_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload OMR sheet image for processing"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Only image files are allowed")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        filename = f"{file_id}{file_extension}"
        file_path = os.path.join(config.UPLOAD_DIR, filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create audit log entry
        audit_log = AuditLog(
            file_id=file_id,
            original_filename=file.filename,
            file_path=file_path,
            exam_id=exam_id,
            student_id=student_id,
            status="uploaded",
            metadata={"file_size": os.path.getsize(file_path)}
        )
        db.add(audit_log)
        db.commit()
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "status": "uploaded",
            "message": "File uploaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/process/{file_id}")
async def process_omr_sheet(
    file_id: str,
    exam_id: str = Form(...),
    student_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Process uploaded OMR sheet"""
    try:
        # Get audit log entry
        audit_log = db.query(AuditLog).filter(AuditLog.file_id == file_id).first()
        if not audit_log:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get exam details
        exam = db.query(Exam).filter(Exam.id == exam_id).first()
        if not exam:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        # Process OMR sheet
        result = omr_processor.process_omr_sheet(
            image_path=audit_log.file_path,
            answer_key=exam.answer_keys,
            exam_id=exam_id,
            student_id=student_id
        )
        
        # Save results to database
        if result['success']:
            # Create or update student if provided
            if student_id:
                student = db.query(Student).filter(Student.student_id == student_id).first()
                if not student:
                    student = Student(
                        student_id=student_id,
                        name=f"Student {student_id}",
                        email=f"{student_id}@university.edu"
                    )
                    db.add(student)
                    db.flush()  # Get the student ID
            
            # Save result
            db_result = ExamResult(
                student_id=student.id if student else None,
                exam_id=exam_id,
                total_score=result['total_score'],
                student_responses=result['answers'],
                confidence_score=result.get('confidence', 0.95),
                processing_status="completed"
            )
            db.add(db_result)
            
            # Update audit log
            audit_log.status = "processed"
            audit_log.result_id = db_result.id
            audit_log.metadata.update({
                "processing_time": result.get('processing_time', 0),
                "confidence": result.get('confidence', 0.95)
            })
            
            db.commit()
            
            return {
                "file_id": file_id,
                "result_id": db_result.id,
                "status": "processed",
                "total_score": result['total_score'],
                "subject_scores": result['subject_scores'],
                "confidence": result.get('confidence', 0.95)
            }
        else:
            # Update audit log with error
            audit_log.status = "failed"
            audit_log.metadata.update({"error": result.get('error', 'Processing failed')})
            db.commit()
            
            raise HTTPException(status_code=422, detail=result.get('error', 'Processing failed'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/results")
async def get_results(
    exam_id: Optional[str] = None,
    student_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get processing results with optional filtering"""
    try:
        query = db.query(ExamResult)
        
        if exam_id:
            query = query.filter(ExamResult.exam_id == exam_id)
        if student_id:
            query = query.filter(ExamResult.student_id == student_id)
        
        results = query.offset(offset).limit(limit).all()
        
        return {
            "results": [
                {
                    "id": result.id,
                    "student_id": result.student_id,
                    "exam_id": result.exam_id,
                    "total_score": result.total_score,
                    "confidence_score": result.confidence_score,
                    "created_at": result.created_at.isoformat()
                }
                for result in results
            ],
            "total": query.count(),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch results: {str(e)}")

@app.get("/results/{result_id}")
async def get_result_detail(result_id: int, db: Session = Depends(get_db)):
    """Get detailed result information"""
    try:
        result = db.query(ExamResult).filter(ExamResult.id == result_id).first()
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        
        return {
            "id": result.id,
            "student_id": result.student_id,
            "exam_id": result.exam_id,
            "total_score": result.total_score,
            "student_responses": result.student_responses,
            "confidence_score": result.confidence_score,
            "processing_status": result.processing_status,
            "created_at": result.created_at.isoformat(),
            "updated_at": result.updated_at.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch result: {str(e)}")

@app.get("/students")
async def get_students(
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get list of students"""
    try:
        students = db.query(Student).offset(offset).limit(limit).all()
        total = db.query(Student).count()
        
        return {
            "students": [
                {
                    "id": student.id,
                    "student_id": student.student_id,
                    "name": student.name,
                    "email": student.email,
                    "phone": student.phone,
                    "created_at": student.created_at.isoformat()
                }
                for student in students
            ],
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch students: {str(e)}")

@app.get("/exams")
async def get_exams(
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get list of exams"""
    try:
        exams = db.query(Exam).offset(offset).limit(limit).all()
        total = db.query(Exam).count()
        
        return {
            "exams": [
                {
                    "id": exam.id,
                    "exam_name": exam.exam_name,
                    "exam_date": exam.exam_date.isoformat(),
                    "total_questions": exam.total_questions,
                    "subjects": exam.subjects,
                    "sheet_versions": exam.sheet_versions,
                    "created_at": exam.created_at.isoformat()
                }
                for exam in exams
            ],
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch exams: {str(e)}")

@app.get("/export/results")
async def export_results(
    format: str = "csv",
    exam_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Export results in various formats"""
    try:
        query = db.query(ExamResult)
        if exam_id:
            query = query.filter(ExamResult.exam_id == exam_id)
        
        results = query.all()
        
        if format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                'Student ID', 'Exam ID', 'Total Score', 'Mathematics', 
                'Physics', 'Chemistry', 'Biology', 'English', 'Confidence', 'Date'
            ])
            
            # Write data
            for result in results:
                writer.writerow([
                    result.student_id,
                    result.exam_id,
                    result.total_score,
                    result.subject_1_score,
                    result.subject_2_score,
                    result.subject_3_score,
                    result.subject_4_score,
                    result.subject_5_score,
                    result.confidence_score,
                    result.created_at.strftime('%Y-%m-%d %H:%M:%S')
                ])
            
            output.seek(0)
            
            # Save to file
            filename = f"results_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(config.EXPORTS_DIR, filename)
            
            with open(filepath, 'w', newline='') as f:
                f.write(output.getvalue())
            
            return FileResponse(
                filepath,
                media_type='text/csv',
                filename=filename
            )
        
        elif format.lower() == "json":
            data = {
                "export_date": datetime.now().isoformat(),
                "total_records": len(results),
                "results": [
                    {
                        "student_id": result.student_id,
                        "exam_id": result.exam_id,
                        "total_score": result.total_score,
                        "subject_scores": {
                            "subject_1": result.subject_1_score,
                            "subject_2": result.subject_2_score,
                            "subject_3": result.subject_3_score,
                            "subject_4": result.subject_4_score,
                            "subject_5": result.subject_5_score
                        },
                        "confidence_score": result.confidence_score,
                        "created_at": result.created_at.isoformat()
                    }
                    for result in results
                ]
            }
            
            filename = f"results_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(config.EXPORTS_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return FileResponse(
                filepath,
                media_type='application/json',
                filename=filename
            )
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv' or 'json'")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/stats")
async def get_statistics(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        total_students = db.query(Student).count()
        total_exams = db.query(Exam).count()
        total_results = db.query(ExamResult).count()
        total_uploads = db.query(AuditLog).count()
        
        # Average scores
        avg_score = db.query(ExamResult).with_entities(
            db.func.avg(ExamResult.total_score)
        ).scalar() or 0
        
        return {
            "total_students": total_students,
            "total_exams": total_exams,
            "total_results": total_results,
            "total_uploads": total_uploads,
            "average_score": round(float(avg_score), 2),
            "system_status": "operational"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch statistics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)