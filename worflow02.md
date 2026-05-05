# Workflow for Auth and Database Files

This document explains at what step each listed file is used and which function or object is used from that file.

## 1. Application Dependencies Installed

**File used:** `backend/requirements.txt`

This file is used before the backend runs. It installs the packages needed by the authentication and database workflow.

Important packages used by these files:

- `fastapi`: creates API routes and dependencies.
- `sqlalchemy`: creates database engine, sessions, and ORM queries.
- `psycopg2-binary`: PostgreSQL driver for synchronous database access.
- `asyncpg`: PostgreSQL driver for async database access.
- `passlib[bcrypt]`: hashes and verifies passwords.
- `bcrypt`: backend used by passlib for bcrypt hashing.
- `python-jose[cryptography]`: creates and verifies JWT tokens.
- `python-multipart`: supports OAuth2 form login data.
- `slowapi`: adds rate limiting to auth endpoints.

## 2. Database Setup

**File used:** `backend/app/database.py`

This file is loaded when another file imports database tools, especially when `router.py` imports `get_db`.

Functions and objects used:

- `engine`: creates the synchronous SQLAlchemy database engine using `settings.DATABASE_URL`.
- `SessionLocal`: creates synchronous database sessions.
- `async_engine`: creates the async SQLAlchemy database engine.
- `AsyncSessionLocal`: creates async database sessions for future async database work.
- `Base`: base class used by ORM models.
- `get_db()`: FastAPI dependency that opens a database session, gives it to the route, and closes it after the request.

Workflow step:

1. `settings.DATABASE_URL` is read.
2. `engine` is created with `create_engine()`.
3. `SessionLocal` is created with `sessionmaker()`.
4. A route asks for `db: Session = Depends(get_db)`.
5. `get_db()` opens a session with `SessionLocal()`.
6. The route uses `db` to query or write data.
7. `get_db()` closes the session in the `finally` block.

## 3. Password and Token Security Setup

**File used:** `backend/app/auth/security.py`

This file is used by auth routes when registering users, logging in users, and creating tokens.

Functions and objects used:

- `_init_bcrypt_backend()`: selects the bcrypt backend when the module loads.
- `pwd_context`: passlib `CryptContext` used for password hashing and verification.
- `verify_password(plain, hashed)`: checks a submitted password against a stored hash.
- `get_password_hash(password)`: converts a plain password into a bcrypt hash.
- `create_access_token(data, expires_delta=None)`: creates a short-lived JWT access token.
- `create_refresh_token(data, expires_delta=None)`: creates a long-lived JWT refresh token.
- `ALGORITHM`: JWT signing algorithm, currently `HS256`.

Workflow step:

1. `security.py` is imported by `router.py`.
2. `_init_bcrypt_backend()` runs once.
3. `pwd_context` is configured for bcrypt hashing.
4. During registration, `get_password_hash()` hashes the user password.
5. During login, `verify_password()` checks the password.
6. During login and refresh, `create_access_token()` creates an access token.
7. During login, `create_refresh_token()` creates a refresh token.

## 4. Register User Workflow

**File used:** `backend/app/auth/router.py`

Route function used:

- `register()`

Other files and functions used inside this route:

- `backend/app/database.py`
  - `get_db()`
- `backend/app/auth/security.py`
  - `get_password_hash()`
- `backend/app/auth/models.py`
  - `User`
- `backend/app/auth/schemas.py`
  - `UserRegister`
  - `UserRead`

Step-by-step flow:

1. Client sends `POST /auth/register`.
2. FastAPI calls `register()`.
3. `Depends(get_db)` calls `get_db()` from `database.py`.
4. `get_db()` creates a database session and passes it as `db`.
5. `register()` checks if username or email already exists:
   - `db.query(User)`
   - `.filter(or_(User.username == user_in.username, User.email == user_in.email))`
   - `.first()`
6. If a user already exists, `register()` raises `HTTPException` with status `400`.
7. If the user is new, `register()` calls `get_password_hash(user_in.password)`.
8. `get_password_hash()` returns a bcrypt password hash.
9. `register()` creates a new `User` object.
10. `db.add(user)` adds it to the session.
11. `db.commit()` saves it to the database.
12. `db.refresh(user)` reloads generated values like `id`.
13. `register()` returns the created user using `UserRead`.
14. `get_db()` closes the database session.

## 5. Login User Workflow

**File used:** `backend/app/auth/router.py`

Route function used:

- `login()`

Other files and functions used inside this route:

- `backend/app/database.py`
  - `get_db()`
- `backend/app/auth/security.py`
  - `verify_password()`
  - `create_access_token()`
  - `create_refresh_token()`
- `backend/app/auth/models.py`
  - `User`
- `backend/app/auth/schemas.py`
  - `TokenResponse`

Step-by-step flow:

1. Client sends `POST /auth/login`.
2. Login data is sent as form data using `OAuth2PasswordRequestForm`.
3. FastAPI calls `login()`.
4. `Depends(get_db)` calls `get_db()` from `database.py`.
5. `get_db()` creates a database session and passes it as `db`.
6. `login()` searches for the user:
   - `db.query(User)`
   - `.filter(User.username == form_data.username)`
   - `.first()`
7. `login()` calls `verify_password(form_data.password, user.hashed_password)`.
8. If the user does not exist or password is wrong, `login()` raises `HTTPException` with status `401`.
9. If credentials are valid, `login()` calls `create_access_token({"sub": user.username}, ...)`.
10. `create_access_token()` creates a JWT with:
    - `sub`
    - `exp`
    - `type: "access"`
11. `login()` calls `create_refresh_token({"sub": user.username}, ...)`.
12. `create_refresh_token()` creates a JWT with:
    - `sub`
    - `exp`
    - `type: "refresh"`
13. `login()` returns `TokenResponse` containing:
    - `access_token`
    - `refresh_token`
    - `token_type: "bearer"`
14. `get_db()` closes the database session.

## 6. Refresh Token Workflow

**File used:** `backend/app/auth/router.py`

Route function used:

- `refresh()`

Other files and functions used inside this route:

- `backend/app/auth/security.py`
  - `ALGORITHM`
  - `create_access_token()`
- `backend/app/auth/schemas.py`
  - `TokenResponse`
- `python-jose`
  - `jwt.decode()`
  - `ExpiredSignatureError`
  - `JWTError`

Step-by-step flow:

1. Client sends `POST /auth/refresh`.
2. Request body contains:
   - `refresh_token`
3. FastAPI calls `refresh()`.
4. `refresh()` decodes the token using `jwt.decode()`.
5. `jwt.decode()` verifies:
   - token signature
   - token expiry
   - signing algorithm
6. `refresh()` checks `payload.get("type")`.
7. If token type is not `"refresh"`, it raises `HTTPException` with status `401`.
8. `refresh()` reads the username from `payload.get("sub")`.
9. If username is missing, it raises `HTTPException` with status `401`.
10. If the token is expired, `ExpiredSignatureError` is caught and `401` is returned.
11. If the token is invalid, `JWTError` is caught and `401` is returned.
12. If token is valid, `refresh()` calls `create_access_token({"sub": username}, ...)`.
13. `create_access_token()` creates a new access token.
14. `refresh()` returns `TokenResponse` with:
    - new `access_token`
    - same `refresh_token`
    - `token_type: "bearer"`

## 7. Shared Dependencies File

**File used:** `backend/app/dependencies.py`

Current status:

- This file currently only contains a docstring.
- No function from this file is used in the listed workflow yet.

Possible future use:

- This file can later hold shared FastAPI dependencies, such as:
  - `get_current_user()`
  - `require_active_user()`
  - `require_admin_user()`

## 8. Complete Auth Flow Summary

### Register

`router.py/register()` -> `database.py/get_db()` -> `security.py/get_password_hash()` -> `User` saved in database.

### Login

`router.py/login()` -> `database.py/get_db()` -> `security.py/verify_password()` -> `security.py/create_access_token()` -> `security.py/create_refresh_token()` -> tokens returned.

### Refresh

`router.py/refresh()` -> `jwt.decode()` -> `security.py/create_access_token()` -> new access token returned.

## 9. File Usage Table

| File | Step Used | Function/Object Used |
| --- | --- | --- |
| `backend/requirements.txt` | Before running backend | Lists packages like `fastapi`, `sqlalchemy`, `passlib`, `python-jose`, `slowapi` |
| `backend/app/database.py` | During routes needing database access | `engine`, `SessionLocal`, `Base`, `get_db()` |
| `backend/app/auth/security.py` | During register, login, and refresh | `_init_bcrypt_backend()`, `verify_password()`, `get_password_hash()`, `create_access_token()`, `create_refresh_token()`, `ALGORITHM` |
| `backend/app/auth/router.py` | Handles auth API requests | `register()`, `login()`, `refresh()` |
| `backend/app/dependencies.py` | Not currently used | No active function yet |

