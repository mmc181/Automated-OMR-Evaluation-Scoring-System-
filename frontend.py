"""
Streamlit Frontend for OMR Evaluation System
"""
import streamlit as st
import requests
import pandas as pd
import json
import os
from datetime import datetime
from PIL import Image
import io

# Optional plotly import
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="OMR Evaluation System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def make_api_request(endpoint, method="GET", data=None, files=None):
    """Make API request to backend"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)

def check_backend_connection():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Sidebar navigation
def sidebar_navigation():
    """Create sidebar navigation"""
    st.sidebar.title("🏠 Navigation")
    
    # Check backend connection
    if check_backend_connection():
        st.sidebar.success("✅ Backend Connected")
    else:
        st.sidebar.error("❌ Backend Disconnected")
        st.sidebar.info("Please start the backend server:\n`python main.py`")
    
    pages = {
        "🏠 Dashboard": "dashboard",
        "📤 Upload OMR Sheets": "upload",
        "👥 Manage Students": "students",
        "📋 Manage Exams": "exams",
        "📊 View Results": "results",
        "🔍 Review Flagged": "review",
        "📈 Analytics": "analytics",
        "⚙️ Settings": "settings"
    }
    
    selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
    return pages[selected_page]

# Dashboard page
def dashboard_page():
    """Main dashboard page"""
    st.markdown('<h1 class="main-header">📝 OMR Evaluation System Dashboard</h1>', unsafe_allow_html=True)
    
    # System statistics
    col1, col2, col3, col4 = st.columns(4)
    
    # Get statistics from API
    students_data, students_error = make_api_request("/students/")
    exams_data, exams_error = make_api_request("/exams/")
    queue_data, queue_error = make_api_request("/processing/queue")
    
    total_students = len(students_data) if students_data and not students_error else 0
    total_exams = len(exams_data) if exams_data and not exams_error else 0
    pending_processing = len([q for q in queue_data if isinstance(q, dict) and q.get('status') == 'queued']) if queue_data and not queue_error and isinstance(queue_data, list) else 0
    
    with col1:
        st.metric("Total Students", total_students)
    with col2:
        st.metric("Total Exams", total_exams)
    with col3:
        st.metric("Pending Processing", pending_processing)
    with col4:
        st.metric("System Status", "🟢 Online" if check_backend_connection() else "🔴 Offline")
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    
    if queue_data:
        df_queue = pd.DataFrame(queue_data)
        if not df_queue.empty:
            # Processing status chart
            status_counts = df_queue['status'].value_counts()
            if PLOTLY_AVAILABLE:
                fig = px.pie(values=status_counts.values, names=status_counts.index, 
                            title="Processing Queue Status")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("**Processing Queue Status:**")
                for status, count in status_counts.items():
                    st.write(f"- {status}: {count}")
        else:
            st.info("No processing queue data available")
    else:
        st.info("No recent activity data available")

# Upload page
def upload_page():
    """File upload page"""
    st.header("📤 Upload OMR Sheets")
    
    # Get available exams
    exams_data, error = make_api_request("/exams/")
    if error:
        st.error(f"Failed to load exams: {error}")
        return
    
    if not exams_data:
        st.warning("No exams available. Please create an exam first.")
        return
    
    # Ensure exams_data is a list
    if not isinstance(exams_data, list):
        st.error(f"Expected list but got {type(exams_data)}: {exams_data}")
        return
    
    # Exam selection
    exam_options = {f"{exam['exam_name']} ({exam['exam_date']})": exam['id'] for exam in exams_data}
    selected_exam_name = st.selectbox("Select Exam", list(exam_options.keys()))
    selected_exam_id = exam_options[selected_exam_name]
    
    # Upload mode selection
    upload_mode = st.radio("Upload Mode", ["Single File", "Batch Upload"])
    
    if upload_mode == "Single File":
        upload_single_file(selected_exam_id)
    else:
        upload_batch_files(selected_exam_id)

def upload_single_file(exam_id):
    """Single file upload"""
    st.subheader("Single File Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        student_id = st.text_input("Student ID (Optional)")
        sheet_version = st.selectbox("Sheet Version", ["A", "B", "C", "D"])
    
    with col2:
        uploaded_file = st.file_uploader(
            "Choose OMR sheet image",
            type=['jpg', 'jpeg', 'png', 'pdf'],
            accept_multiple_files=False
        )
    
    if uploaded_file and st.button("Upload and Process"):
        with st.spinner("Uploading and processing..."):
            files = {"file": uploaded_file}
            data = {
                "exam_id": exam_id,
                "student_id": student_id,
                "sheet_version": sheet_version
            }
            
            response, error = make_api_request("/upload/single", "POST", data=data, files=files)
            
            if error:
                st.error(f"Upload failed: {error}")
            else:
                st.success("File uploaded successfully!")
                st.json(response)

def upload_batch_files(exam_id):
    """Batch file upload"""
    st.subheader("Batch File Upload")
    
    uploaded_files = st.file_uploader(
        "Choose OMR sheet images",
        type=['jpg', 'jpeg', 'png', 'pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Upload Batch"):
        with st.spinner(f"Uploading {len(uploaded_files)} files..."):
            files = [("files", file) for file in uploaded_files]
            data = {"exam_id": exam_id}
            
            response, error = make_api_request("/upload/batch", "POST", data=data, files=files)
            
            if error:
                st.error(f"Batch upload failed: {error}")
            else:
                st.success(f"Successfully uploaded {len(uploaded_files)} files!")
                st.json(response)

# Students management page
def students_page():
    """Students management page"""
    st.header("👥 Manage Students")
    
    tab1, tab2 = st.tabs(["View Students", "Add Student"])
    
    with tab1:
        students_data, error = make_api_request("/students/")
        if error:
            st.error(f"Failed to load students: {error}")
        elif students_data:
            df = pd.DataFrame(students_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No students found")
    
    with tab2:
        st.subheader("Add New Student")
        with st.form("add_student"):
            student_id = st.text_input("Student ID*", placeholder="e.g., STU001")
            name = st.text_input("Full Name*", placeholder="e.g., John Doe")
            email = st.text_input("Email", placeholder="e.g., john@example.com")
            phone = st.text_input("Phone", placeholder="e.g., +1234567890")
            
            if st.form_submit_button("Add Student"):
                if student_id and name:
                    data = {
                        "student_id": student_id,
                        "name": name,
                        "email": email if email else None,
                        "phone": phone if phone else None
                    }
                    
                    response, error = make_api_request("/students/", "POST", data=data)
                    if error:
                        st.error(f"Failed to add student: {error}")
                    else:
                        st.success("Student added successfully!")
                        st.rerun()
                else:
                    st.error("Student ID and Name are required")

# Exams management page
def exams_page():
    """Exams management page"""
    st.header("📋 Manage Exams")
    
    tab1, tab2 = st.tabs(["View Exams", "Create Exam"])
    
    with tab1:
        exams_data, error = make_api_request("/exams/")
        if error:
            st.error(f"Failed to load exams: {error}")
        elif exams_data:
            for exam in exams_data:
                with st.expander(f"{exam['exam_name']} - {exam['exam_date']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Total Questions:** {exam['total_questions']}")
                        st.write(f"**Subjects:** {', '.join(exam['subjects'].keys()) if exam['subjects'] else 'Not set'}")
                    with col2:
                        st.write(f"**Sheet Versions:** {', '.join(exam['sheet_versions']) if exam['sheet_versions'] else 'Not set'}")
                        st.write(f"**Created:** {exam['created_at']}")
        else:
            st.info("No exams found")
    
    with tab2:
        st.subheader("Create New Exam")
        with st.form("create_exam"):
            exam_name = st.text_input("Exam Name*", placeholder="e.g., Midterm Exam 2024")
            exam_date = st.date_input("Exam Date*")
            total_questions = st.number_input("Total Questions", min_value=1, value=100)
            
            st.write("**Subjects Configuration:**")
            subjects = {}
            for i in range(5):
                col1, col2 = st.columns(2)
                with col1:
                    subject_name = st.text_input(f"Subject {i+1} Name", placeholder=f"Subject {i+1}")
                with col2:
                    subject_questions = st.number_input(f"Questions", min_value=0, value=20, key=f"q_{i}")
                
                if subject_name:
                    subjects[subject_name] = subject_questions
            
            sheet_versions = st.multiselect("Sheet Versions", ["A", "B", "C", "D"], default=["A"])
            
            if st.form_submit_button("Create Exam"):
                if exam_name and exam_date:
                    data = {
                        "exam_name": exam_name,
                        "exam_date": exam_date.isoformat(),
                        "total_questions": total_questions,
                        "subjects": subjects,
                        "sheet_versions": sheet_versions,
                        "answer_keys": {version: {} for version in sheet_versions}
                    }
                    
                    response, error = make_api_request("/exams/", "POST", data=data)
                    if error:
                        st.error(f"Failed to create exam: {error}")
                    else:
                        st.success("Exam created successfully!")
                        st.rerun()
                else:
                    st.error("Exam name and date are required")

# Results page
def results_page():
    """Results viewing page"""
    st.header("📊 View Results")
    
    # Get exams for filtering
    exams_data, _ = make_api_request("/exams/")
    if not exams_data:
        st.warning("No exams available")
        return
    
    # Exam selection
    exam_options = {f"{exam['exam_name']} ({exam['exam_date']})": exam['id'] for exam in exams_data}
    selected_exam_name = st.selectbox("Select Exam", list(exam_options.keys()))
    selected_exam_id = exam_options[selected_exam_name]
    
    # Get results for selected exam
    results_data, error = make_api_request(f"/results/exam/{selected_exam_id}")
    
    if error:
        st.error(f"Failed to load results: {error}")
        return
    
    if not results_data:
        st.info("No results found for this exam")
        return
    
    # Display results
    df = pd.DataFrame(results_data)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Submissions", len(df))
    with col2:
        st.metric("Average Score", f"{df['total_score'].mean():.2f}")
    with col3:
        st.metric("Highest Score", f"{df['total_score'].max():.2f}")
    with col4:
        st.metric("Lowest Score", f"{df['total_score'].min():.2f}")
    
    # Results table
    st.subheader("Detailed Results")
    st.dataframe(df, use_container_width=True)
    
    # Export options
    st.subheader("Export Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export CSV"):
            response, error = make_api_request(f"/export/exam/{selected_exam_id}/csv")
            if not error:
                st.success("CSV export ready for download")
    
    with col2:
        if st.button("Export Excel"):
            response, error = make_api_request(f"/export/exam/{selected_exam_id}/excel")
            if not error:
                st.success("Excel export ready for download")
    
    with col3:
        if st.button("Export JSON"):
            response, error = make_api_request(f"/export/exam/{selected_exam_id}/json")
            if not error:
                st.success("JSON export ready for download")

# Review flagged results page
def review_page():
    """Review flagged results page"""
    st.header("🔍 Review Flagged Results")
    
    flagged_data, error = make_api_request("/results/flagged")
    
    if error:
        st.error(f"Failed to load flagged results: {error}")
        return
    
    if not flagged_data:
        st.info("No flagged results found")
        return
    
    st.write(f"Found {len(flagged_data)} flagged results requiring review")
    
    for result in flagged_data:
        with st.expander(f"Student: {result.get('student_id', 'Unknown')} - Confidence: {result.get('confidence_score', 0):.2f}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Result Details:**")
                st.json(result)
            
            with col2:
                st.write("**Flagged Questions:**")
                if result.get('flagged_questions'):
                    for question in result['flagged_questions']:
                        st.write(f"- Question {question}")
                else:
                    st.write("No specific questions flagged")

# Analytics page
def analytics_page():
    """Analytics and reporting page"""
    st.header("📈 Analytics")
    
    # Get all results for analytics
    exams_data, _ = make_api_request("/exams/")
    if not exams_data:
        st.warning("No exam data available for analytics")
        return
    
    # Exam selection for analytics
    exam_options = {f"{exam['exam_name']} ({exam['exam_date']})": exam['id'] for exam in exams_data}
    selected_exam_name = st.selectbox("Select Exam for Analytics", list(exam_options.keys()))
    selected_exam_id = exam_options[selected_exam_name]
    
    # Get exam statistics
    stats_data, error = make_api_request(f"/exams/{selected_exam_id}/statistics")
    
    if error:
        st.error(f"Failed to load statistics: {error}")
        return
    
    if stats_data:
        # Score distribution
        st.subheader("Score Distribution")
        results_data, _ = make_api_request(f"/results/exam/{selected_exam_id}")
        
        if results_data:
            df = pd.DataFrame(results_data)
            
            # Histogram of total scores
            if PLOTLY_AVAILABLE:
                fig = px.histogram(df, x='total_score', nbins=20, title="Total Score Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("**Total Score Distribution:**")
                st.write(f"- Average Score: {df['total_score'].mean():.2f}")
                st.write(f"- Median Score: {df['total_score'].median():.2f}")
                st.write(f"- Standard Deviation: {df['total_score'].std():.2f}")
            
            # Subject-wise performance
            st.subheader("Subject-wise Performance")
            subject_cols = [col for col in df.columns if col.startswith('subject_') and col.endswith('_score')]
            
            if subject_cols:
                subject_data = df[subject_cols].mean()
                if PLOTLY_AVAILABLE:
                    fig = px.bar(x=subject_data.index, y=subject_data.values, title="Average Subject Scores")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("**Average Subject Scores:**")
                    for subject, score in subject_data.items():
                        st.write(f"- {subject}: {score:.2f}")

# Settings page
def settings_page():
    """System settings page"""
    st.header("⚙️ System Settings")
    
    st.subheader("API Configuration")
    new_api_url = st.text_input("Backend API URL", value=API_BASE_URL)
    
    if st.button("Test Connection"):
        try:
            response = requests.get(f"{new_api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Connection successful!")
            else:
                st.error(f"❌ Connection failed: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
    
    st.subheader("System Information")
    st.info(f"""
    - Frontend Version: 1.0.0
    - Backend URL: {API_BASE_URL}
    - Connection Status: {'🟢 Connected' if check_backend_connection() else '🔴 Disconnected'}
    """)

# Main application
def main():
    """Main application function"""
    # Sidebar navigation
    current_page = sidebar_navigation()
    
    # Route to appropriate page
    if current_page == "dashboard":
        dashboard_page()
    elif current_page == "upload":
        upload_page()
    elif current_page == "students":
        students_page()
    elif current_page == "exams":
        exams_page()
    elif current_page == "results":
        results_page()
    elif current_page == "review":
        review_page()
    elif current_page == "analytics":
        analytics_page()
    elif current_page == "settings":
        settings_page()

if __name__ == "__main__":
    main()