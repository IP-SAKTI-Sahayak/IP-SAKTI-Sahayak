"""Password hashing and signed bearer-token authentication."""

import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend import db


ROOT = Path(__file__).resolve().parent.parent
SECRET_PATH = ROOT / "data" / "runtime" / "auth_secret.key"
bearer_scheme = HTTPBearer(auto_error=False)


def signing_secret():
    configured = os.getenv("AUTH_SECRET_KEY")
    if configured:
        return configured
    SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with SECRET_PATH.open("x", encoding="ascii") as handle:
            handle.write(secrets.token_urlsafe(48))
    except FileExistsError:
        pass
    return SECRET_PATH.read_text(encoding="ascii").strip()


def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("ascii")


def verify_password(password, encoded_hash):
    try:
        return bcrypt.checkpw(password.encode("utf-8"), encoded_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False


def issue_token(user):
    now = datetime.now(timezone.utc)
    expires = now + timedelta(seconds=int(os.getenv("AUTH_TOKEN_TTL_SECONDS", "28800")))
    return jwt.encode(
        {"sub": str(user["id"]), "role": user["role"], "iat": now, "exp": expires, "iss": "ip-sakti-sahayak"},
        signing_secret(),
        algorithm="HS256",
    )


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authentication required.", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(
            credentials.credentials,
            signing_secret(),
            algorithms=["HS256"],
            issuer="ip-sakti-sahayak",
            options={"require": ["sub", "role", "iat", "exp", "iss"]},
        )
        user_id = int(payload["sub"])
    except (jwt.InvalidTokenError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid or expired authentication token.", headers={"WWW-Authenticate": "Bearer"})
    user = db.get_user_by_id(user_id)
    if user is None or user["role"] != payload.get("role"):
        raise HTTPException(status_code=401, detail="Invalid or expired authentication token.", headers={"WWW-Authenticate": "Bearer"})
    return db.public_user(user)


def require_roles(*roles):
    def dependency(user=Depends(get_current_user)):
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Insufficient role.")
        return user
    return dependency
