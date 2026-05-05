"""
security.py — Password hashing and JWT creation.

All cryptographic operations for the auth system live here:
  - bcrypt (via passlib) for password hashing — one-way, salted, slow by design.
  - HS256 JWT (via python-jose) for stateless session tokens — signed with
    SECRET_KEY so the server can verify them without a DB lookup.

Why bcrypt with 12 rounds?
  bcrypt is deliberately slow (each extra round doubles the computation time).
  12 rounds ~= 300 ms on modern hardware, which is negligible for a login
  request but makes brute-force attacks impractical.

Why HS256 JWTs?
  HS256 is a symmetric algorithm — the same SECRET_KEY signs and verifies the
  token.  It is simpler than RS256 (asymmetric) and sufficient for a single-
  service backend where only one server ever verifies tokens.
"""

from datetime import datetime, timedelta, timezone
import os
from jose import jwt
from passlib.context import CryptContext
from passlib.handlers.bcrypt import bcrypt as bcrypt_handler
from app.auth.config import auth_settings

# JWT algorithm — all tokens are signed/verified with this.
ALGORITHM = "HS256"


def _init_bcrypt_backend() -> None:
    """
    Select the best available bcrypt C extension for passlib.

    passlib supports multiple bcrypt backends:
      - "bcrypt"  — the fast C extension (bcrypt PyPI package).  Preferred.
      - "builtin" — pure-Python fallback, slower but always available.

    The try/except attempts to activate the C extension.  If it isn't
    installed (e.g. on a minimal Docker image), it falls back to the
    built-in pure-Python version by setting the PASSLIB_BUILTIN_BCRYPT
    environment variable, which passlib reads at import time.

    This function is called once at module load so every subsequent
    verify_password / get_password_hash call uses the pre-selected backend.
    """
    try:
        bcrypt_handler.set_backend("bcrypt")        # prefer fast C extension
    except Exception:
        os.environ.setdefault("PASSLIB_BUILTIN_BCRYPT", "1")
        bcrypt_handler.set_backend("builtin")       # fallback to pure Python


# Run the backend selection when this module is first imported.
_init_bcrypt_backend()

# CryptContext is passlib's unified hashing interface.
# schemes=["bcrypt"]   — only bcrypt is accepted for new hashes.
# deprecated="auto"    — if a stored hash uses an older/weaker scheme,
#                        verify() still works but reports it as deprecated
#                        (useful for future algorithm migrations).
# bcrypt__rounds=12    — work factor; higher = slower (12 ≈ 300 ms on modern CPU).
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


def verify_password(plain: str, hashed: str) -> bool:
    """
    Check whether a plain-text password matches a stored bcrypt hash.

    How it works:
      bcrypt embeds the salt inside the hash string, so pwd_context.verify()
      extracts the salt, re-hashes the plain password with that salt, and
      compares the result in constant time (preventing timing attacks).

    Args:
        plain  : The password the user typed at login.
        hashed : The bcrypt hash stored in auth_users.hashed_password.

    Returns:
        True if the password matches the hash, False otherwise.
        Never raises — a wrong password returns False, not an exception.
    """
    return pwd_context.verify(plain, hashed)


def get_password_hash(password: str) -> str:
    """
    Hash a plain-text password with bcrypt for safe storage.

    The hash includes:
      - The bcrypt algorithm identifier ($2b$)
      - The work factor (12 rounds)
      - A random 128-bit salt (generated fresh every call)
      - The derived key

    The salt is different on every call, so two users with the same password
    will have completely different hashes — preventing rainbow-table attacks.

    Args:
        password : The plain-text password from the registration form.

    Returns:
        A 60-character bcrypt hash string safe to store in the database.
    """
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Create a short-lived JWT access token.

    The access token is presented by the frontend on every API call via
    the Authorization: Bearer <token> header.  It expires quickly (30 min
    by default) to limit the damage if it's stolen.

    Payload structure:
      {
        "sub": "username",          ← the subject (who the token belongs to)
        "exp": <unix timestamp>,    ← expiry (python-jose validates this automatically)
        "type": "access"            ← prevents refresh tokens being used as access tokens
      }

    Args:
        data         : Dict containing at least {"sub": username}.
        expires_delta: Override the default expiry from auth_settings.
                       Pass timedelta(minutes=N) to set a custom expiry.

    Returns:
        A signed HS256 JWT string.
    """
    to_encode = data.copy()   # don't mutate the caller's dict
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=auth_settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "type": "access"})
    # jwt.encode signs the payload with SECRET_KEY using HS256
    return jwt.encode(to_encode, auth_settings.SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Create a long-lived JWT refresh token.

    The refresh token is stored in localStorage and is ONLY sent to
    POST /auth/refresh — it never goes on regular API calls.  This reduces
    the window in which it can be stolen from network traffic.

    It lives for 7 days by default, giving the user a "stay logged in"
    experience without keeping the short-lived access token valid for long.

    The "type": "refresh" claim ensures that if someone tries to use a
    refresh token as an access token, the dependency in dependencies.py
    will reject it (it checks for type == "access").

    Args:
        data         : Dict containing at least {"sub": username}.
        expires_delta: Override the default 7-day expiry.

    Returns:
        A signed HS256 JWT string with a longer expiry than access tokens.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(days=auth_settings.REFRESH_TOKEN_EXPIRE_DAYS)
    )
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, auth_settings.SECRET_KEY, algorithm=ALGORITHM)
