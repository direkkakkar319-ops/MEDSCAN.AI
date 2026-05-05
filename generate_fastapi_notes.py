"""
Generate docs/FastAPI_Notes.pdf  -- FastAPI notes with MEDSCAN.AI examples.
Run:  python generate_fastapi_notes.py
"""

import os
from fpdf import FPDF

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLUE  = (30,  80, 160)
DBLUE = (15,  45, 100)
GREEN = (20, 130,  70)
GRAY  = (245, 245, 245)
LGRAY = (200, 200, 200)
BLACK = (30,  30,  30)
WHITE = (255, 255, 255)
RED   = (180,  30,  30)


class PDF(FPDF):
    def header(self):
        pass  # custom headers per section

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*LGRAY)
        self.cell(0, 6, f"MEDSCAN.AI -- FastAPI Notes  |  Page {self.page_no()}", align="C")
        self.set_text_color(*BLACK)

    # ---- title page --------------------------------------------------------
    def title_page(self, title, subtitle):
        self.add_page()
        # full-page blue background
        self.set_fill_color(*BLUE)
        self.rect(0, 0, 210, 297, "F")
        # white panel
        self.set_fill_color(*WHITE)
        self.rect(20, 60, 170, 130, "F")
        # title
        self.set_xy(20, 80)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(*DBLUE)
        self.multi_cell(170, 12, title, align="C")
        # subtitle
        self.set_font("Helvetica", "", 13)
        self.set_text_color(*BLACK)
        self.set_xy(20, 130)
        self.multi_cell(170, 7, subtitle, align="C")
        # footer note
        self.set_xy(20, 175)
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(*WHITE)
        self.multi_cell(170, 6, "FastAPI basics explained side-by-side with MEDSCAN.AI project usage.", align="C")
        self.set_text_color(*BLACK)

    # ---- section heading ---------------------------------------------------
    def h1(self, text):
        self.ln(4)
        self.set_fill_color(*BLUE)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 9, "  " + text, fill=True, ln=True)
        self.set_text_color(*BLACK)
        self.ln(2)

    # ---- sub-heading -------------------------------------------------------
    def h2(self, text):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*DBLUE)
        self.cell(0, 7, text, ln=True)
        self.set_text_color(*BLACK)
        self.ln(1)

    # ---- body text ---------------------------------------------------------
    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*BLACK)
        self.multi_cell(0, 5, text)
        self.ln(1)

    # ---- code block --------------------------------------------------------
    def code(self, lines, label=""):
        self.ln(1)
        if label:
            self.set_font("Helvetica", "B", 8)
            self.set_fill_color(*DBLUE)
            self.set_text_color(*WHITE)
            self.cell(0, 5, "  " + label, fill=True, ln=True)
            self.set_text_color(*BLACK)
        self.set_fill_color(*GRAY)
        self.set_font("Courier", "", 8)
        self.set_text_color(40, 40, 40)
        for line in lines:
            self.cell(0, 4.5, "  " + line, fill=True, ln=True)
        self.set_text_color(*BLACK)
        self.ln(2)

    # ---- two-column comparison table ---------------------------------------
    def table(self, rows, col_a="General FastAPI Concept", col_b="MEDSCAN.AI Usage"):
        self.ln(2)
        w = 90
        # header row
        self.set_fill_color(*DBLUE)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 9)
        self.cell(w, 7, "  " + col_a, fill=True, border=1)
        self.cell(w, 7, "  " + col_b, fill=True, border=1, ln=True)
        # data rows
        for i, (a, b) in enumerate(rows):
            fill_color = GRAY if i % 2 == 0 else WHITE
            self.set_fill_color(*fill_color)
            self.set_text_color(*BLACK)
            self.set_font("Helvetica", "", 9)
            # measure height needed
            lines_a = self._wrap(a, w - 4)
            lines_b = self._wrap(b, w - 4)
            n = max(len(lines_a), len(lines_b), 1)
            row_h = 5 * n
            x, y = self.get_x(), self.get_y()
            # column A
            self.rect(x, y, w, row_h, "F")
            self.set_xy(x + 2, y + 1)
            for ln_a in lines_a:
                self.cell(w - 4, 5, ln_a)
                self.set_xy(x + 2, self.get_y() + 5)
            # column B
            self.set_xy(x + w, y)
            self.rect(x + w, y, w, row_h, "F")
            self.set_xy(x + w + 2, y + 1)
            for ln_b in lines_b:
                self.cell(w - 4, 5, ln_b)
                self.set_xy(x + w + 2, self.get_y() + 5)
            # border around both cells
            self.set_draw_color(*LGRAY)
            self.rect(x, y, w, row_h)
            self.rect(x + w, y, w, row_h)
            self.set_xy(x, y + row_h)
        self.ln(3)

    def _wrap(self, text, max_w):
        """Split text into lines that fit within max_w mm (approx 2.8 chars/mm at size 9)."""
        words = text.split()
        lines, current = [], ""
        for word in words:
            test = (current + " " + word).strip()
            if self.get_string_width(test) <= max_w:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]

    # ---- bullet list -------------------------------------------------------
    def bullets(self, items):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*BLACK)
        for item in items:
            self.cell(6, 5, "  *")
            self.multi_cell(0, 5, item)
        self.ln(1)


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

def build(pdf):
    # ========================================================================
    # SECTION 1: What is FastAPI?
    # ========================================================================
    pdf.add_page()
    pdf.h1("1. What Is FastAPI?")
    pdf.body(
        "FastAPI is a modern Python web framework for building HTTP APIs. "
        "It uses Python type hints to auto-validate request/response data and "
        "auto-generates interactive docs (Swagger UI at /docs, ReDoc at /redoc). "
        "It is built on top of Starlette (ASGI) and Pydantic."
    )
    pdf.table([
        ("A web framework that maps URL paths to Python functions called 'route handlers'.",
         "MEDSCAN creates the FastAPI app in app/main.py: app = FastAPI(title='MEDSCAN.AI')"),
        ("Every route handler returns data; FastAPI serializes it to JSON automatically.",
         "Routes like GET /api/reports return Python dicts/Pydantic models -> JSON."),
        ("Auto-generates /docs (Swagger UI) so you can test endpoints in the browser.",
         "Developers test /api/upload, /api/status/{id} directly in the browser during dev."),
        ("Uses ASGI (async), so it handles many simultaneous requests efficiently.",
         "Concurrent report uploads and status polls are handled without blocking."),
    ])
    pdf.code([
        "# Minimal FastAPI application",
        "from fastapi import FastAPI",
        "app = FastAPI()",
        "",
        "@app.get('/hello')",
        "def hello():",
        "    return {'message': 'Hello, world'}",
    ], label="Concept: minimal app")
    pdf.code([
        "# MEDSCAN.AI -- app/main.py",
        "from fastapi import FastAPI",
        "from app.routers import auth, reports, ml",
        "",
        "app = FastAPI(",
        "    title='MEDSCAN.AI',",
        "    description='Medical report risk analysis API',",
        "    version='1.0.0',",
        ")",
        "app.include_router(auth.router,    prefix='/auth')",
        "app.include_router(reports.router, prefix='/api')",
        "app.include_router(ml.router,      prefix='/api')",
    ], label="MEDSCAN.AI: app/main.py")

    # ========================================================================
    # SECTION 2: Routers
    # ========================================================================
    pdf.add_page()
    pdf.h1("2. Routers -- Splitting Routes Across Files")
    pdf.body(
        "As an API grows, keeping every route in main.py becomes unmanageable. "
        "FastAPI's APIRouter lets you define routes in separate files and then "
        "mount them all into the main app with include_router(). "
        "Each router can have its own prefix, tags, and dependencies."
    )
    pdf.table([
        ("APIRouter() creates a mini-app that can hold its own routes.",
         "MEDSCAN has routers: auth.py, reports.py, ml.py, compare.py."),
        ("prefix='/auth' means all routes inside become /auth/login, /auth/register etc.",
         "auth.router has prefix='/auth', so POST /auth/login, POST /auth/register."),
        ("tags=['Auth'] groups the routes under a named section in /docs.",
         "reports.router uses tags=['Reports'] so docs are neatly grouped."),
        ("dependencies=[Depends(x)] applies x to EVERY route in the router.",
         "reports.router passes Depends(get_current_active_user) to protect all report routes."),
    ])
    pdf.code([
        "# app/routers/reports.py",
        "from fastapi import APIRouter, Depends",
        "from app.dependencies import get_current_active_user",
        "",
        "router = APIRouter(",
        "    prefix='/api',",
        "    tags=['Reports'],",
        "    dependencies=[Depends(get_current_active_user)],",
        ")",
        "",
        "@router.get('/reports')",
        "def list_reports(...):",
        "    ...",
    ], label="MEDSCAN.AI: routers/reports.py")

    # ========================================================================
    # SECTION 3: Path & Query Parameters
    # ========================================================================
    pdf.add_page()
    pdf.h1("3. Path Parameters and Query Parameters")
    pdf.body(
        "Path parameters are part of the URL itself: /reports/{report_id}. "
        "Query parameters follow a '?' in the URL: /reports?limit=10&offset=0. "
        "FastAPI reads Python function arguments and maps them automatically -- "
        "if the name matches a path variable it is a path param; otherwise a query param."
    )
    pdf.table([
        ("{report_id} in the path string -> int report_id arg -> FastAPI validates it is an int.",
         "GET /api/status/{report_id} -> def get_status(report_id: int, db: Session)."),
        ("Query params are optional if given a default value (limit: int = 10).",
         "GET /api/reports can accept ?limit=20 to control how many reports are returned."),
        ("FastAPI returns HTTP 422 automatically if a param fails type validation.",
         "Sending /api/status/abc returns 422 Unprocessable Entity with a clear error message."),
    ])
    pdf.code([
        "# Concept: path param + query param",
        "@app.get('/items/{item_id}')",
        "def get_item(item_id: int, q: str = None):",
        "    return {'id': item_id, 'q': q}",
        "",
        "# URL: /items/42?q=search",
        "# -> item_id=42, q='search'",
    ], label="Concept: params")
    pdf.code([
        "# MEDSCAN.AI -- routers/ml.py",
        "@router.get('/status/{report_id}')",
        "def get_status(",
        "    report_id: int,",
        "    db: Session = Depends(get_db),",
        "    current_user: User = Depends(get_current_active_user),",
        "):",
        "    report = db.query(Report).filter(",
        "        Report.id == report_id,",
        "        Report.user_id == current_user.id,",
        "    ).first()",
        "    if not report:",
        "        raise HTTPException(status_code=404)",
        "    return report",
    ], label="MEDSCAN.AI: GET /api/status/{report_id}")

    # ========================================================================
    # SECTION 4: Request Body and Pydantic
    # ========================================================================
    pdf.add_page()
    pdf.h1("4. Request Body and Pydantic Schemas")
    pdf.body(
        "When a client sends JSON in the request body (POST/PUT), FastAPI reads it "
        "and validates it against a Pydantic model. If validation fails, FastAPI "
        "returns HTTP 422 with a detailed error. Pydantic models also serve as "
        "response_model to filter what gets sent back to the client."
    )
    pdf.table([
        ("class LoginRequest(BaseModel): email: str; password: str",
         "AuthRequest model receives email + password for POST /auth/login."),
        ("response_model=TokenResponse tells FastAPI to filter output through that schema.",
         "POST /auth/login returns {access_token, refresh_token, token_type} -- nothing else."),
        ("Field(...) = required; Field(None) or Optional[str] = optional.",
         "ReportResponse has Optional fields for result/extracted_metrics (null while processing)."),
        ("Validators (@field_validator) run before the route handler is called.",
         "Email fields are validated for correct format before reaching auth logic."),
    ])
    pdf.code([
        "# Concept: Pydantic request + response model",
        "from pydantic import BaseModel",
        "",
        "class LoginRequest(BaseModel):",
        "    email: str",
        "    password: str",
        "",
        "class TokenResponse(BaseModel):",
        "    access_token: str",
        "    token_type: str",
        "",
        "@app.post('/login', response_model=TokenResponse)",
        "def login(body: LoginRequest):",
        "    # FastAPI auto-parses JSON body -> LoginRequest",
        "    ...",
    ], label="Concept: Pydantic models")

    # ========================================================================
    # SECTION 5: Dependency Injection
    # ========================================================================
    pdf.add_page()
    pdf.h1("5. Dependency Injection with Depends()")
    pdf.body(
        "Depends() is FastAPI's built-in dependency injection system. "
        "You write a function that returns something (a DB session, a user object, etc.) "
        "and declare it as a default argument using Depends(). FastAPI calls the "
        "dependency function automatically before calling your route handler. "
        "Dependencies can call other dependencies, forming a chain."
    )
    pdf.table([
        ("Depends(get_db) -> FastAPI calls get_db(), injects the Session into the route.",
         "Every DB-touching route in MEDSCAN uses db: Session = Depends(get_db)."),
        ("get_db() is a generator (yield) so cleanup (session.close()) runs after the response.",
         "get_db() opens a SessionLocal, yields it, then closes in a finally block."),
        ("Dependencies chain: get_current_user calls Depends(oauth2_scheme) internally.",
         "get_current_active_user -> get_current_user -> oauth2_scheme -> Authorization header."),
        ("Depends() on a router-level applies the dep to every route in that router.",
         "All /api/reports routes are protected because reports.router uses Depends(get_current_active_user)."),
    ])
    pdf.code([
        "# MEDSCAN.AI dependency chain",
        "",
        "# Step 1: database session",
        "def get_db():",
        "    db = SessionLocal()",
        "    try:",
        "        yield db",
        "    finally:",
        "        db.close()",
        "",
        "# Step 2: extract JWT from Authorization header",
        "oauth2_scheme = OAuth2PasswordBearer(tokenUrl='/auth/login')",
        "",
        "# Step 3: decode JWT -> user object",
        "def get_current_user(",
        "    token: str = Depends(oauth2_scheme),",
        "    db: Session = Depends(get_db),",
        "):",
        "    payload = decode_jwt(token)  # raises 401 if invalid",
        "    return db.query(User).get(payload['sub'])",
        "",
        "# Step 4: check user is active",
        "def get_current_active_user(",
        "    user: User = Depends(get_current_user),",
        "):",
        "    if not user.is_active:",
        "        raise HTTPException(403)",
        "    return user",
    ], label="MEDSCAN.AI: full dependency chain")

    # ========================================================================
    # SECTION 6: File Uploads
    # ========================================================================
    pdf.add_page()
    pdf.h1("6. File Uploads with UploadFile")
    pdf.body(
        "FastAPI handles multipart/form-data uploads through the UploadFile class. "
        "The client sends a form-data POST with the file attached. FastAPI "
        "reads the file into memory (or streams it) and exposes it as an "
        "UploadFile object. You can read the bytes, check the filename and "
        "content type, and then save or process the file."
    )
    pdf.table([
        ("File(...) declares the parameter as a required upload field.",
         "POST /api/upload uses file: UploadFile = File(...)."),
        ("UploadFile.read() returns bytes; UploadFile.filename gives the original name.",
         "MEDSCAN reads the bytes and uploads them to Supabase S3 via boto3."),
        ("Content-Type header must NOT be set manually; the browser sets it with the boundary.",
         "Frontend api.js note: do not set Content-Type for FormData uploads."),
        ("File validation (size, MIME type) must be done manually inside the route.",
         "MEDSCAN checks file.content_type in ['image/jpeg','image/png','application/pdf']."),
    ])
    pdf.code([
        "# MEDSCAN.AI -- routers/reports.py",
        "from fastapi import UploadFile, File",
        "",
        "@router.post('/upload')",
        "async def upload_report(",
        "    file: UploadFile = File(...),",
        "    report_type: str = Form(...),",
        "    db: Session = Depends(get_db),",
        "    current_user: User = Depends(get_current_active_user),",
        "):",
        "    contents = await file.read()  # bytes",
        "    # upload to Supabase S3",
        "    s3 = _get_s3()",
        "    s3.put_object(Bucket=BUCKET, Key=filename, Body=contents)",
        "    # create DB record",
        "    report = Report(user_id=current_user.id, filename=filename, ...)",
        "    db.add(report); db.commit()",
        "    # dispatch Celery task",
        "    process_medical_report.delay(report.id, filename, report_type)",
        "    return {'report_id': report.id, 'status': 'processing'}",
    ], label="MEDSCAN.AI: POST /api/upload")

    # ========================================================================
    # SECTION 7: HTTPException and Error Handling
    # ========================================================================
    pdf.add_page()
    pdf.h1("7. HTTPException and Error Responses")
    pdf.body(
        "Raise HTTPException to return an error response immediately and stop "
        "executing the route handler. FastAPI converts it to a JSON response "
        "with the status_code and a {detail: ...} body. You can also add "
        "custom headers (e.g. WWW-Authenticate for 401 errors)."
    )
    pdf.table([
        ("raise HTTPException(status_code=404, detail='Not found') -> {detail: 'Not found'}",
         "Report not found or not owned by user -> 404 with detail message."),
        ("raise HTTPException(401, headers={'WWW-Authenticate': 'Bearer'})",
         "get_current_user raises 401 when token is missing, expired, or tampered."),
        ("422 Unprocessable Entity is raised automatically by FastAPI on Pydantic failures.",
         "Sending wrong JSON to POST /auth/register returns 422 with field-level errors."),
        ("You can register @app.exception_handler(exc_type) for global custom handling.",
         "MEDSCAN uses FastAPI's default handlers; no custom exception handlers yet."),
    ])
    pdf.code([
        "# Concept: HTTPException usage",
        "from fastapi import HTTPException",
        "",
        "@app.get('/reports/{id}')",
        "def get_report(id: int, db: Session = Depends(get_db)):",
        "    report = db.query(Report).get(id)",
        "    if not report:",
        "        raise HTTPException(status_code=404, detail='Report not found')",
        "    return report",
        "",
        "# Response body: {\"detail\": \"Report not found\"}",
        "# HTTP status: 404",
    ], label="Concept: HTTPException")

    # ========================================================================
    # SECTION 8: JWT Authentication
    # ========================================================================
    pdf.add_page()
    pdf.h1("8. JWT Authentication -- Login and Token Flow")
    pdf.body(
        "JWT (JSON Web Token) is a signed string that proves who the caller is. "
        "The server creates it at login and signs it with a secret key. "
        "On every subsequent request the client sends the JWT in the "
        "Authorization: Bearer <token> header. FastAPI reads and verifies it "
        "to identify the user -- no database lookup needed for every request."
    )
    pdf.table([
        ("Login: POST /auth/login -> server creates and returns access_token + refresh_token.",
         "POST /auth/login returns {access_token (30 min), refresh_token (7 days)}."),
        ("access_token is short-lived (minutes). Stored in localStorage on the frontend.",
         "Frontend stores access_token in localStorage; attaches it to every API call."),
        ("refresh_token is long-lived (days). Used to get a new access_token silently.",
         "POST /auth/refresh -> new access_token. apiFetch() retries automatically on 401."),
        ("OAuth2PasswordBearer extracts the Bearer token from the Authorization header.",
         "oauth2_scheme = OAuth2PasswordBearer(tokenUrl='/auth/login') used in get_current_user."),
    ])
    pdf.code([
        "# MEDSCAN.AI -- auth flow",
        "",
        "# On login:",
        "def create_tokens(user_id: int):",
        "    access  = jwt.encode({'sub': str(user_id), 'exp': now + 30min}, SECRET)",
        "    refresh = jwt.encode({'sub': str(user_id), 'exp': now + 7days}, SECRET)",
        "    return access, refresh",
        "",
        "# On every protected request:",
        "def get_current_user(token = Depends(oauth2_scheme)):",
        "    try:",
        "        payload = jwt.decode(token, SECRET, algorithms=['HS256'])",
        "        user_id = payload['sub']",
        "    except jwt.ExpiredSignatureError:",
        "        raise HTTPException(401, 'Token expired')",
        "    return db.query(User).get(user_id)",
        "",
        "# Frontend silent refresh (api.js):",
        "# if response.status == 401:",
        "#   POST /auth/refresh -> new access_token -> retry request",
    ], label="MEDSCAN.AI: JWT flow")

    # ========================================================================
    # SECTION 9: Background Tasks with Celery
    # ========================================================================
    pdf.add_page()
    pdf.h1("9. Background Task Processing with Celery")
    pdf.body(
        "OCR and ML inference take several seconds -- too long to block an HTTP response. "
        "FastAPI's built-in BackgroundTasks is simple but runs in the same process. "
        "MEDSCAN uses Celery + Redis instead: the route handler dispatches a task and "
        "returns immediately; a separate worker process runs the task in the background. "
        "The client polls GET /api/status/{report_id} until status changes to 'completed'."
    )
    pdf.table([
        ("process_medical_report.delay(id, filename, type) dispatches and returns a task_id.",
         "POST /api/upload calls .delay() and immediately returns {report_id, status: processing}."),
        ("Celery worker picks up the task from Redis (the broker queue).",
         "One Celery worker process runs with concurrency=1 to avoid GPU memory conflicts."),
        ("Task result is stored in Redis for 1 hour (result_expires=3600).",
         "Status polls check the DB record (not Celery result) for report.status field."),
        ("@shared_task(bind=True) gives the task access to self.retry() for retries.",
         "compare_reports retries up to 3 times with exponential back-off on ML service errors."),
    ])
    pdf.code([
        "# MEDSCAN.AI -- task_queue/tasks.py",
        "",
        "@shared_task(bind=True, max_retries=3)",
        "def process_medical_report(self, report_id, filename, report_type):",
        "    try:",
        "        # 1. download file from S3",
        "        # 2. POST file to ML microservice",
        "        # 3. save OCR results to DB",
        "        # 4. save prediction to DB",
        "        # 5. mark report.status = 'completed'",
        "        pass",
        "    except Exception as exc:",
        "        raise self.retry(exc=exc, countdown=2 ** self.request.retries)",
        "",
        "# Route handler:",
        "@router.post('/upload')",
        "def upload(...):",
        "    process_medical_report.delay(report.id, filename, report_type)",
        "    return {'report_id': report.id, 'status': 'processing'}",
    ], label="MEDSCAN.AI: Celery task dispatch")

    # ========================================================================
    # SECTION 10: Database with SQLAlchemy + FastAPI
    # ========================================================================
    pdf.add_page()
    pdf.h1("10. Database Integration with SQLAlchemy")
    pdf.body(
        "FastAPI has no built-in ORM, but SQLAlchemy (the most popular Python ORM) "
        "integrates naturally via the Depends() pattern. A SessionLocal factory "
        "creates one DB connection per request. The get_db() generator yields the "
        "session to the route and closes it in a finally block after the response."
    )
    pdf.table([
        ("create_engine(DATABASE_URL) creates the connection pool to PostgreSQL.",
         "MEDSCAN connects to Neon PostgreSQL using DATABASE_URL from environment."),
        ("SessionLocal = sessionmaker(bind=engine) is the session factory.",
         "get_db() creates a SessionLocal, yields it, closes it when done."),
        ("Base = declarative_base() is the parent class for all model classes.",
         "User, Report, Prediction, Comparison all extend Base."),
        ("db.query(Model).filter(...).first() is how you query a single row.",
         "get_status uses db.query(Report).filter(Report.id==id, Report.user_id==uid).first()."),
    ])
    pdf.code([
        "# MEDSCAN.AI -- app/database.py",
        "from sqlalchemy import create_engine",
        "from sqlalchemy.orm import sessionmaker, declarative_base",
        "",
        "engine = create_engine(DATABASE_URL, pool_pre_ping=True)",
        "SessionLocal = sessionmaker(bind=engine)",
        "Base = declarative_base()",
        "",
        "def get_db():",
        "    db = SessionLocal()",
        "    try:",
        "        yield db",
        "    finally:",
        "        db.close()",
    ], label="MEDSCAN.AI: database.py")

    # ========================================================================
    # SECTION 11: CORS
    # ========================================================================
    pdf.add_page()
    pdf.h1("11. CORS -- Allowing the React Frontend to Call the API")
    pdf.body(
        "Browsers block cross-origin requests by default (CORS policy). "
        "Because the React frontend is served from a different origin than the "
        "FastAPI backend (different domain or port), CORS headers must be added "
        "to every response. FastAPI's CORSMiddleware handles this automatically."
    )
    pdf.table([
        ("add_middleware(CORSMiddleware, allow_origins=[...]) adds CORS headers to responses.",
         "MEDSCAN allows the Vercel frontend URL and localhost:5173 (Vite dev server)."),
        ("allow_credentials=True is required when the client sends cookies or Authorization headers.",
         "Frontend sends Authorization: Bearer token, so credentials=True is needed."),
        ("allow_methods=['*'] permits GET, POST, PUT, DELETE, OPTIONS, etc.",
         "MEDSCAN uses GET, POST, DELETE so ['*'] covers all of them."),
        ("The browser sends a preflight OPTIONS request first; CORSMiddleware handles it.",
         "No special handling needed -- middleware responds to OPTIONS automatically."),
    ])
    pdf.code([
        "# MEDSCAN.AI -- app/main.py",
        "from fastapi.middleware.cors import CORSMiddleware",
        "",
        "app.add_middleware(",
        "    CORSMiddleware,",
        "    allow_origins=[",
        "        'https://medscan-ai.vercel.app',",
        "        'http://localhost:5173',",
        "    ],",
        "    allow_credentials=True,",
        "    allow_methods=['*'],",
        "    allow_headers=['*'],",
        ")",
    ], label="MEDSCAN.AI: CORS setup")

    # ========================================================================
    # SECTION 12: Response Models and Status Codes
    # ========================================================================
    pdf.add_page()
    pdf.h1("12. Response Models and HTTP Status Codes")
    pdf.body(
        "By default FastAPI returns HTTP 200 and whatever the route function returns. "
        "You can override both with decorator arguments. response_model filters "
        "which fields are sent to the client (useful for hiding internal fields). "
        "status_code sets the HTTP status code for successful responses."
    )
    pdf.table([
        ("@app.post('/users', status_code=201) returns 201 Created on success.",
         "POST /auth/register returns 201 when a new user account is created."),
        ("response_model=UserOut strips password_hash and other internal fields.",
         "UserOut schema exposes id, email, full_name -- not password_hash."),
        ("response_model_exclude_none=True removes null fields from the JSON output.",
         "ReportResponse uses this so null result/metrics are omitted while processing."),
        ("Return Response(status_code=204) for endpoints that return no body.",
         "DELETE /api/reports/{id} returns 204 No Content after deletion."),
    ])
    pdf.code([
        "# Concept: status code + response model",
        "@app.post('/users', status_code=201, response_model=UserOut)",
        "def create_user(body: UserCreate, db: Session = Depends(get_db)):",
        "    user = User(email=body.email, ...)",
        "    db.add(user); db.commit()",
        "    return user  # FastAPI filters through UserOut schema",
        "",
        "# Only fields in UserOut are sent -- password_hash is excluded",
    ], label="Concept: response model")

    # ========================================================================
    # SECTION 13: Environment Variables and Settings
    # ========================================================================
    pdf.add_page()
    pdf.h1("13. Environment Variables and Settings Management")
    pdf.body(
        "Hard-coding secrets (DB passwords, JWT secret keys, API keys) in source "
        "code is a security risk. FastAPI projects typically use pydantic-settings "
        "or python-dotenv to load config from environment variables or a .env file. "
        "The settings object is created once and injected via Depends()."
    )
    pdf.table([
        ("class Settings(BaseSettings): DATABASE_URL: str reads from env automatically.",
         "MEDSCAN loads DATABASE_URL, SECRET_KEY, REDIS_URL, S3 keys from environment."),
        ("pydantic-settings validates types and raises on startup if required vars are missing.",
         "Missing DATABASE_URL causes the app to crash on startup, not silently at query time."),
        ("In production (Render.com), vars are set in the dashboard, not in .env.",
         "MEDSCAN uses Render environment groups so secrets never appear in source code."),
        ("python-dotenv loads a .env file in local development automatically.",
         ".env.example documents required vars; the actual .env is in .gitignore."),
    ])
    pdf.code([
        "# MEDSCAN.AI -- config approach",
        "import os",
        "from dotenv import load_dotenv",
        "load_dotenv()  # reads .env in local dev",
        "",
        "DATABASE_URL = os.getenv('DATABASE_URL')   # Neon PostgreSQL",
        "SECRET_KEY   = os.getenv('SECRET_KEY')     # JWT signing key",
        "REDIS_URL    = os.getenv('REDIS_URL')      # Upstash Redis",
        "S3_BUCKET    = os.getenv('SUPABASE_BUCKET')",
        "",
        "# In production these come from Render environment variables",
        "# -- never from a committed .env file",
    ], label="MEDSCAN.AI: environment config")

    # ========================================================================
    # SECTION 14: Async vs Sync Route Handlers
    # ========================================================================
    pdf.add_page()
    pdf.h1("14. Async vs Sync Route Handlers")
    pdf.body(
        "FastAPI supports both async def (non-blocking) and def (blocking, run in "
        "a thread pool) route handlers. Use async def when you await I/O operations "
        "(async DB drivers, async HTTP clients). Use def when you use synchronous "
        "libraries like SQLAlchemy (sync) or requests, since FastAPI runs them in a "
        "thread pool automatically to avoid blocking the event loop."
    )
    pdf.table([
        ("async def route(): await some_async_call() -- non-blocking I/O.",
         "upload() uses async def because await file.read() is async (UploadFile)."),
        ("def route(): -- FastAPI runs this in a thread pool, keeping the event loop free.",
         "get_reports(), get_status() use def because SQLAlchemy is synchronous."),
        ("Do NOT use time.sleep() in async routes -- use asyncio.sleep() instead.",
         "Celery tasks are separate processes, so sleeping there does not affect FastAPI."),
        ("Mixing sync SQLAlchemy with async FastAPI is fine -- FastAPI runs sync in threads.",
         "MEDSCAN uses sync SQLAlchemy throughout; no async DB driver is needed."),
    ])
    pdf.code([
        "# async def: file read is awaitable (UploadFile is async)",
        "@router.post('/upload')",
        "async def upload_report(file: UploadFile = File(...), ...):",
        "    contents = await file.read()  # non-blocking",
        "    ...",
        "",
        "# def: SQLAlchemy is sync, FastAPI runs in thread pool",
        "@router.get('/reports')",
        "def list_reports(db: Session = Depends(get_db), ...):",
        "    return db.query(Report).filter(...).all()  # sync query",
    ], label="MEDSCAN.AI: async vs sync")

    # ========================================================================
    # SECTION 15: ML Microservice Communication
    # ========================================================================
    pdf.add_page()
    pdf.h1("15. Calling the ML Microservice from Celery")
    pdf.body(
        "MEDSCAN runs the ML pipeline (OCR + prediction) as a separate microservice "
        "on Render.com. The Celery task sends the uploaded file to the ML service "
        "via HTTP POST (multipart/form-data) and receives the prediction result as JSON. "
        "This decouples the API from the GPU-heavy ML workload."
    )
    pdf.table([
        ("The Celery task uses requests.post() to call the ML service URL.",
         "process_medical_report sends the S3 file to ML_SERVICE_URL/process."),
        ("The ML service (FastAPI too) returns OCR text + risk predictions as JSON.",
         "Response: {extracted_metrics: {...}, risks: {...}, risk_level: 'moderate'}."),
        ("ML_SERVICE_URL is set as an environment variable so it can differ per env.",
         "Dev: http://localhost:8001, Production: the Render ML service URL."),
        ("Retries on network errors prevent a transient blip from failing the report.",
         "Celery task retries up to 3 times with exponential back-off (2^retry seconds)."),
    ])
    pdf.code([
        "# task_queue/tasks.py -- calling the ML microservice",
        "import requests",
        "",
        "ML_SERVICE_URL = os.getenv('ML_SERVICE_URL')",
        "",
        "@shared_task(bind=True, max_retries=3)",
        "def process_medical_report(self, report_id, filename, report_type):",
        "    # download file from S3",
        "    s3 = boto3.client('s3', ...)",
        "    obj = s3.get_object(Bucket=BUCKET, Key=filename)",
        "    file_bytes = obj['Body'].read()",
        "",
        "    # POST to ML microservice",
        "    resp = requests.post(",
        "        f'{ML_SERVICE_URL}/process',",
        "        files={'file': (filename, file_bytes)},",
        "        data={'report_type': report_type},",
        "        timeout=120,",
        "    )",
        "    resp.raise_for_status()",
        "    result = resp.json()",
        "",
        "    # save to DB",
        "    _save_ocr_results(report_id, result['extracted_metrics'])",
        "    _save_prediction(report_id, result)",
    ], label="MEDSCAN.AI: ML microservice call")

    # ========================================================================
    # SECTION 16: Startup and Shutdown Events
    # ========================================================================
    pdf.add_page()
    pdf.h1("16. Startup and Shutdown Lifecycle Events")
    pdf.body(
        "FastAPI lets you register functions that run when the app starts up "
        "and when it shuts down. This is useful for warming up caches, creating "
        "DB tables, loading ML models into memory, or closing connections cleanly."
    )
    pdf.table([
        ("@app.on_event('startup') runs once when the server first starts.",
         "MEDSCAN creates DB tables (Base.metadata.create_all) on startup."),
        ("@app.on_event('shutdown') runs when the server is stopping.",
         "Connection pools and caches can be cleaned up on shutdown."),
        ("In newer FastAPI, lifespan context manager replaces on_event (recommended).",
         "MEDSCAN uses the older on_event style; lifespan is the future direction."),
        ("OCR model warm-up: loading PaddleOCR on first request adds latency.",
         "get_ocr_runner() is called at startup to pre-load the OCR singleton."),
    ])
    pdf.code([
        "# MEDSCAN.AI -- app/main.py startup",
        "from app.database import engine, Base",
        "",
        "@app.on_event('startup')",
        "def startup():",
        "    Base.metadata.create_all(bind=engine)  # create tables if missing",
        "",
        "# Modern lifespan alternative (FastAPI 0.93+):",
        "from contextlib import asynccontextmanager",
        "",
        "@asynccontextmanager",
        "async def lifespan(app):",
        "    Base.metadata.create_all(bind=engine)  # startup",
        "    yield",
        "    # shutdown logic here",
        "",
        "app = FastAPI(lifespan=lifespan)",
    ], label="MEDSCAN.AI: startup event")

    # ========================================================================
    # SECTION 17: Quick Route Reference
    # ========================================================================
    pdf.add_page()
    pdf.h1("17. MEDSCAN.AI Quick Route Reference")
    pdf.body("All routes in the MEDSCAN.AI API grouped by category.")
    pdf.table([
        ("POST /auth/register",          "Create new user account. Returns 201 + user object."),
        ("POST /auth/login",             "Authenticate user. Returns access_token + refresh_token."),
        ("POST /auth/refresh",           "Exchange refresh_token for a new access_token."),
        ("POST /api/upload",             "Upload medical report image/PDF. Returns report_id + 'processing'."),
        ("GET  /api/reports",            "List all reports for the logged-in user."),
        ("GET  /api/status/{report_id}", "Poll processing status and retrieve prediction result."),
        ("DELETE /api/reports/{id}",     "Delete a report and its associated predictions."),
        ("POST /api/compare",            "Compare two reports. Returns trend data + delta risks."),
        ("GET  /api/compare/{id}",       "Retrieve a previously run comparison result."),
        ("GET  /api/history",            "Paginated history of all past reports with results."),
    ], col_a="Route", col_b="Description")

    # ========================================================================
    # SECTION 18: End-to-End Request Lifecycle
    # ========================================================================
    pdf.add_page()
    pdf.h1("18. End-to-End Request Lifecycle: GET /api/reports")
    pdf.body(
        "Tracing a single request from browser to database and back illustrates "
        "how all the FastAPI concepts connect in practice."
    )

    steps = [
        ("Browser", "React calls apiFetch('/api/reports') with Authorization: Bearer <token>."),
        ("CORS",    "CORSMiddleware checks the Origin header and adds Access-Control-Allow-Origin."),
        ("Router",  "FastAPI matches GET /api/reports to list_reports() in reports.py."),
        ("Auth",    "Depends(get_current_active_user) -> Depends(get_current_user) -> decodes JWT."),
        ("DB",      "Depends(get_db) opens a SQLAlchemy session for this request."),
        ("Handler", "list_reports() queries db.query(Report).filter(user_id==...).all()."),
        ("Serial.", "FastAPI serializes the list of Report objects through ReportResponse schema."),
        ("Response","HTTP 200 JSON array is sent back; get_db() finally block closes the session."),
    ]

    for i, (stage, desc) in enumerate(steps, 1):
        self_ref = pdf
        y = pdf.get_y()
        self_ref.set_fill_color(*BLUE)
        self_ref.set_text_color(*WHITE)
        self_ref.set_font("Helvetica", "B", 9)
        self_ref.cell(28, 7, f"  {i}. {stage}", fill=True, border=0)
        self_ref.set_fill_color(*GRAY)
        self_ref.set_text_color(*BLACK)
        self_ref.set_font("Helvetica", "", 9)
        self_ref.cell(0, 7, "  " + desc, fill=True, border=0, ln=True)
        if i < len(steps):
            self_ref.set_draw_color(*LGRAY)
            self_ref.set_xy(10, pdf.get_y())

    pdf.ln(5)
    pdf.body(
        "Key takeaway: FastAPI, Depends(), SQLAlchemy, and Pydantic each handle "
        "one layer of the request. You write only the business logic in the handler; "
        "everything else is handled by the framework."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)

    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=10, top=10, right=10)
    pdf.set_text_color(*BLACK)

    pdf.title_page(
        "FastAPI Notes",
        "General Concepts  x  MEDSCAN.AI Project Usage\n"
        "A side-by-side reference guide",
    )
    build(pdf)

    out = os.path.join("docs", "FastAPI_Notes.pdf")
    pdf.output(out)
    print(f"PDF written to: {os.path.abspath(out)}")
