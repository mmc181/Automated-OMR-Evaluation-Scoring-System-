"""
Pydantic schemas for API request/response models
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime


# Student schemas
class StudentBase(BaseModel):
    student_id: str = Field(..., description="Unique student identifier")
    name: str = Field(..., description="Student full name")
    email: Optional[str] = Field(None, description="Student email address")
    phone: Optional[str] = Field(None, description="Student phone number")


class StudentCreate(StudentBase):
    pass


class StudentResponse(StudentBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Exam schemas
class ExamBase(BaseModel):
    exam_name: str = Field(..., description="Name of the exam")
    exam_date: datetime = Field(..., description="Date of the exam")
    total_questions: int = Field(100, description="Total number of questions")
    subjects: Dict[str, Any] = Field(..., description="Subject configuration")
    sheet_versions: List[str] = Field(..., description="Available sheet versions")
    answer_keys: Dict[str, Dict[str, str]] = Field(..., description="Answer keys for each version")


class ExamCreate(ExamBase):
    pass


class ExamResponse(ExamBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Exam Result schemas
class ExamResultBase(BaseModel):
    student_id: int
    exam_id: int
    sheet_version: str
    subject_1_score: float = 0.0
    subject_2_score: float = 0.0
    subject_3_score: float = 0.0
    subject_4_score: float = 0.0
    subject_5_score: float = 0.0
    total_score: float = 0.0
    student_responses: Optional[Dict[str, str]] = None
    correct_answers: Optional[Dict[str, str]] = None
    processing_status: str = "pending"
    confidence_score: float = 0.0
    flagged_questions: Optional[List[str]] = None


class ExamResultCreate(ExamResultBase):
    pass


class ExamResultResponse(ExamResultBase):
    id: int
    processed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Processing schemas
class ProcessingRequest(BaseModel):
    exam_id: Optional[int] = None
    student_id: Optional[str] = None
    sheet_version: Optional[str] = None


class ProcessingResponse(BaseModel):
    processing_id: str
    status: str
    message: str


class ProcessingStatus(BaseModel):
    processing_id: str
    status: str
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Upload schemas
class UploadResponse(BaseModel):
    success: bool
    message: str
    file_id: Optional[str] = None
    processing_id: Optional[str] = None


# Statistics schemas
class SubjectStatistics(BaseModel):
    subject_name: str
    average_score: float
    max_score: float
    min_score: float
    pass_rate: float


class ExamStatistics(BaseModel):
    exam_id: int
    exam_name: str
    total_students: int
    average_total_score: float
    subject_statistics: List[SubjectStatistics]
    score_distribution: Dict[str, int]


# Export schemas
class ExportRequest(BaseModel):
    exam_id: int
    format: str = Field(..., pattern="^(json|csv|excel)$")
    include_details: bool = True


class ExportResponse(BaseModel):
    success: bool
    download_url: str
    filename: str
    expires_at: datetime