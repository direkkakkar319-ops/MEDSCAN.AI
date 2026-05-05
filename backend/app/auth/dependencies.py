"""
dependencies.py — FastAPI authentication dependencies.

These two functions are injected into protected routes via Depends().
They form a two-step guard:

  1. get_current_user()        — decodes the JWT, looks up the user in the DB.
  2. get_current_active_user() — additionally rejects inactive accounts.

Usage in a route:
    @router.get("/protected")
    def my_route(current_user: User = Depends(get_current_active_user)):
        ...

FastAPI automatically calls the dependency before the route handler and
passes the returned User object in.  If the dependency raises HTTPException,
FastAPI returns the error response immediately without running the route.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import ExpiredSignatureError, JWTError, jwt
from sqlalchemy.orm import Session
from app.auth.config import auth_settings
from app.auth.models import User
from app.database import get_db
from app.auth.security import ALGORITHM

# OAuth2PasswordBearer tells FastAPI that this API uses Bearer token auth.
# tokenUrl="/auth/login" is used by the Swagger UI (/docs) to know where
# to send the login form when the user clicks "Authorize".
# In actual requests it extracts the token from the Authorization header:
#   Authorization: Bearer <jwt>
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Decode the JWT access token and load the corresponding User from the DB.

    Flow:
      1. FastAPI extracts the raw JWT string from the Authorization header
         via oauth2_scheme (raises 401 automatically if the header is missing).
      2. python-jose decodes and verifies the signature against SECRET_KEY.
         It also validates the 'exp' claim — expired tokens raise
         ExpiredSignatureError before we even read the payload.
      3. We check that token_type == "access" to prevent refresh tokens
         (type="refresh") from being used to authenticate API calls.
      4. We extract the 'sub' claim (the username) and look up the user row.

    Raises:
        401 UNAUTHORIZED — if the token is missing, expired, malformed,
                           wrong type, or the user no longer exists in the DB.

    Returns:
        The User ORM object for the authenticated user.
    """
    try:
        # Decode verifies the signature AND the expiry claim automatically.
        payload = jwt.decode(token, auth_settings.SECRET_KEY, algorithms=[ALGORITHM])

        # Guard: reject refresh tokens used as access tokens.
        token_type = payload.get("type")
        if token_type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )

        # 'sub' (subject) holds the username set during create_access_token().
        username: str | None = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )

    except ExpiredSignatureError as exc:
        # Token was valid but the 'exp' timestamp is in the past.
        # The frontend should catch this 401 and call /auth/refresh.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        ) from exc
    except JWTError as exc:
        # Any other JWT problem: bad signature, invalid base64, unknown algo…
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        ) from exc

    # Look up the user in the database to ensure they still exist.
    # (A deleted user's token would still decode successfully otherwise.)
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Extend get_current_user() with an active-account check.

    is_active is set to True on registration.  An admin could set it to
    False to disable an account without deleting it (e.g. for policy
    violations, spam, etc.).

    Most protected routes use this dependency rather than get_current_user()
    directly so that disabled accounts are automatically blocked everywhere.

    Raises:
        403 FORBIDDEN — if the user's account has been deactivated.

    Returns:
        The same User object from get_current_user() if the account is active.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user
