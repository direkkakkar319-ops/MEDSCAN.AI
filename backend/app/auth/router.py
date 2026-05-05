"""
auth/router.py — Authentication API routes.

Three endpoints handle the complete auth lifecycle:

  POST /auth/register  — create a new user account
  POST /auth/login     — exchange credentials for JWT tokens
  POST /auth/refresh   — exchange a refresh token for a new access token

Rate limits (slowapi per IP):
  /register : 3/minute  — prevents bulk account creation
  /login    : 5/minute  — slows brute-force password guessing
  /refresh  : 10/minute — refresh is called silently by the frontend on 401s

Token strategy:
  - Access token  : 30 min TTL, sent on every API request
  - Refresh token : 7 day TTL, only sent to /auth/refresh
  The separation limits the blast radius of a stolen access token to 30 min.
"""

from datetime import timedelta
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import or_
from sqlalchemy.orm import Session
from app.auth.models import User
from app.auth.schemas import TokenResponse, UserRead, UserRegister
from app.auth.security import (
    ALGORITHM,
    create_access_token,
    create_refresh_token,
    get_password_hash,
    verify_password,
)
from app.auth.config import auth_settings
from app.database import get_db

router = APIRouter(prefix="/auth", tags=["auth"])

# Module-level limiter instance — limits are declared per-route below.
limiter = Limiter(key_func=get_remote_address)


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
@limiter.limit("3/minute")
def register(
    request: Request,  # required by @limiter.limit — slowapi reads the client IP from it
    user_in: UserRegister,
    db: Session = Depends(get_db),
) -> UserRead:
    """
    Create a new user account.

    Request body (JSON):
        { "username": str, "email": str, "password": str }

    Steps:
      1. Check whether the username OR email is already taken.
         A single OR query avoids two round-trips and gives a clear error.
      2. Hash the password with bcrypt (12 rounds) — the plain text is
         NEVER stored.
      3. Insert a new User row with is_active=True.
      4. Return the created user (without the password hash) via UserRead.

    Note: The frontend currently sends username = email (both fields are
    the same value) to keep the login form simple (email only).

    Raises:
        400 BAD REQUEST  — username or email already registered.
        422 UNPROCESSABLE — request body fails Pydantic validation
                            (missing fields, invalid email format, etc.).
    """
    # Check for conflict before hashing to avoid the expensive bcrypt call
    # when the user already exists.
    existing = (
        db.query(User)
        .filter(or_(User.username == user_in.username, User.email == user_in.email))
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered",
        )

    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),  # bcrypt hash
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)   # re-fetch from DB to populate auto-generated fields (id, created_at)
    return user


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
def login(
    request: Request,  # required by @limiter.limit — slowapi reads the client IP from it
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> TokenResponse:
    """
    Verify user credentials and return a pair of JWT tokens.

    Request format: application/x-www-form-urlencoded
        username=<email>&password=<password>

    Why form data instead of JSON?
      OAuth2PasswordRequestForm is the standard OAuth2 password flow format.
      FastAPI's Swagger UI (/docs) auto-generates the login form for it.
      The frontend sends URLSearchParams which encodes as form data.

    Steps:
      1. Look up user by username (which the frontend sends as the email).
      2. Verify the submitted password against the stored bcrypt hash.
         Both checks are combined so the response time is identical whether
         the user doesn't exist or the password is wrong — preventing
         username enumeration via timing.
      3. Create a short-lived access token and a long-lived refresh token.
      4. Return both tokens in the response body.

    Raises:
        401 UNAUTHORIZED — wrong username or password.
    """
    user = db.query(User).filter(User.username == form_data.username).first()

    # Combine existence check and password check into one condition.
    # This prevents timing attacks: both branches take roughly the same time
    # because verify_password always runs the bcrypt hash computation.
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    # Build the JWT payload — "sub" (subject) is the standard claim for
    # identifying the principal (user).  We use username, not id, so the
    # token is human-readable and doesn't expose numeric IDs.
    access_token = create_access_token(
        {"sub": user.username},
        expires_delta=timedelta(minutes=auth_settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh_token = create_refresh_token(
        {"sub": user.username},
        expires_delta=timedelta(days=auth_settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",   # standard OAuth2 token type
    )


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit("10/minute")
def refresh(
    request: Request,
    refresh_token: str = Body(..., embed=True),  # expects { "refresh_token": "..." }
) -> TokenResponse:
    """
    Exchange a valid refresh token for a new access token.

    Called automatically by the frontend's apiFetch() wrapper when any API
    call returns 401.  The user never sees this — it's a silent background
    request that extends their session.

    Request body (JSON):
        { "refresh_token": "<JWT>" }

    Steps:
      1. Decode and verify the refresh token signature and expiry.
      2. Check that token_type == "refresh" — access tokens are rejected here
         to prevent misuse.
      3. Issue a fresh access token with a new 30-minute expiry.
      4. Return the same refresh_token — it keeps its original expiry.
         If the refresh token has also expired, the frontend must redirect
         to the login page.

    Raises:
        401 UNAUTHORIZED — token expired, invalid signature, or wrong type.

    Note: jose is imported inside the function to avoid circular imports
    at module load time in some edge cases.
    """
    from jose import ExpiredSignatureError, JWTError, jwt

    try:
        payload = jwt.decode(
            refresh_token,
            auth_settings.SECRET_KEY,
            algorithms=[ALGORITHM],
        )

        # Reject access tokens sent to this endpoint by mistake.
        token_type = payload.get("type")
        if token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )

        username: str | None = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )

    except ExpiredSignatureError as exc:
        # The refresh token itself has expired (7-day window passed).
        # The frontend must redirect to the login page.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        ) from exc
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        ) from exc

    # Issue a brand-new access token for the same user.
    access_token = create_access_token(
        {"sub": username},
        expires_delta=timedelta(minutes=auth_settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    # Return the SAME refresh_token — its expiry is unchanged.
    # Only issue a new refresh token if implementing refresh token rotation.
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )
