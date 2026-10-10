"""
TrustShield AI — Authentication & Role-Based Authorization Framework.

Production-grade security module providing:
- Configurable authentication (API Key header or Bearer Token)
- Role-Based Access Control (RBAC: Admin, Operator, Analyst)
- Safe local development mode toggle
- Production enforcement safeguards (fails closed in production)
- Redacted audit logging (zero secret exposure)
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Callable, Optional

from fastapi import Depends, Header, HTTPException, Query, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Security schemes
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)


class Role(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    ANALYST = "analyst"


class AuthUser(BaseModel):
    username: str
    role: Role
    is_authenticated: bool = True


def is_auth_enabled() -> bool:
    """
    Check if authentication is enforced.
    Fails closed:
    - If TRUSTSHIELD_ENV or ENV is 'production', auth is MANDATORY. Attempting to disable raises RuntimeError.
    - If TRUSTSHIELD_AUTH_ENABLED is explicitly set to 'true'/'1', auth is active.
    - In dev/test without explicit setting, auth is optional for development velocity.
    """
    env = os.getenv("TRUSTSHIELD_ENV", os.getenv("ENV", "development")).strip().lower()
    is_prod = env == "production"

    explicit = os.getenv("TRUSTSHIELD_AUTH_ENABLED")
    if explicit is not None:
        enabled = explicit.lower() in ("true", "1", "yes")
        if is_prod and not enabled:
            raise RuntimeError(
                "CRITICAL SECURITY: Cannot disable authentication when running in production environment."
            )
        return enabled

    return is_prod


def get_configured_keys() -> dict[str, tuple[str, Role]]:
    """
    Resolve configured API keys to (username, Role).
    In production, missing keys raise RuntimeError to prevent running with insecure defaults.
    """
    env = os.getenv("TRUSTSHIELD_ENV", os.getenv("ENV", "development")).strip().lower()
    is_prod = env == "production"

    admin_key = os.getenv("TRUSTSHIELD_ADMIN_KEY")
    operator_key = os.getenv("TRUSTSHIELD_API_KEY")

    if is_prod:
        if not admin_key or len(admin_key) < 16:
            raise RuntimeError(
                "CRITICAL SECURITY: TRUSTSHIELD_ADMIN_KEY must be set with at least 16 characters in production."
            )
        if not operator_key or len(operator_key) < 16:
            raise RuntimeError(
                "CRITICAL SECURITY: TRUSTSHIELD_API_KEY must be set with at least 16 characters in production."
            )

    # Development defaults (only active when NOT in production)
    admin_key = admin_key or "dev-admin-secret-key-32chars-long!"
    operator_key = operator_key or "dev-operator-api-key-32chars-long"

    return {
        admin_key: ("admin_user", Role.ADMIN),
        operator_key: ("operator_user", Role.OPERATOR),
    }


def validate_security_configuration() -> None:
    """
    Validate that all critical security configurations are safe at startup.
    Fails startup closed if insecure configurations are detected in production.
    """
    env = os.getenv("TRUSTSHIELD_ENV", os.getenv("ENV", "development")).strip().lower()
    if env == "production":
        # 1. Enforce authentication cannot be bypassed
        if not is_auth_enabled():
            raise RuntimeError("CRITICAL SECURITY: Authentication must be enabled in production.")

        # 2. Enforce strong, explicit API keys
        get_configured_keys()

        # 3. Enforce safe CORS whitelist
        raw_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
        origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
        if not origins or "*" in origins:
            raise RuntimeError(
                "CRITICAL SECURITY: Explicit CORS_ALLOWED_ORIGINS must be configured in production without wildcards."
            )

        # 4. Enforce explicit Neo4j credentials
        neo_pw = os.getenv("NEO4J_PASSWORD", "")
        if not neo_pw or neo_pw in ("password", "trustshield_secret", "neo4j"):
            raise RuntimeError(
                "CRITICAL SECURITY: Strong, non-default NEO4J_PASSWORD must be configured in production."
            )


def authenticate_request(
    api_key: Optional[str] = Security(_api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
    api_key_query: Optional[str] = Query(default=None, alias="api_key"),
) -> AuthUser:
    """
    Authenticate an incoming request via API Key header, Bearer token, or query parameter.
    Returns AuthUser or raises HTTP 401.
    """
    if not is_auth_enabled():
        # In unauthenticated dev/test mode
        return AuthUser(username="dev_unauthenticated", role=Role.ADMIN, is_authenticated=False)

    token = None
    if api_key:
        token = api_key.strip()
    elif bearer and bearer.credentials:
        token = bearer.credentials.strip()
    elif api_key_query:
        token = api_key_query.strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide 'X-API-Key' header or Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    keys = get_configured_keys()
    if token in keys:
        username, role = keys[token]
        return AuthUser(username=username, role=role, is_authenticated=True)

    logger.warning("Failed authentication attempt with invalid key.")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(min_role: Role) -> Callable[[AuthUser], AuthUser]:
    """
    Dependency factory verifying that authenticated user has required role hierarchy.
    Hierarchy: ADMIN (3) > OPERATOR (2) > ANALYST (1).
    """
    role_hierarchy = {
        Role.ANALYST: 1,
        Role.OPERATOR: 2,
        Role.ADMIN: 3,
    }

    def role_checker(user: AuthUser = Depends(authenticate_request)) -> AuthUser:
        if not is_auth_enabled():
            return user

        user_level = role_hierarchy.get(user.role, 0)
        required_level = role_hierarchy.get(min_role, 0)

        if user_level < required_level:
            logger.warning(
                "Access denied for user '%s' (role: %s) requiring %s.",
                user.username,
                user.role.value,
                min_role.value,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires '{min_role.value}' privilege level.",
            )
        return user

    return role_checker
