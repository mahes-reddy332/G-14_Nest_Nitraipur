
from fastapi import Header, HTTPException, Depends
from typing import Optional
from core.security.rbac import User, UserRole, RBACService, Permission

# Mock user database
MOCK_USERS = {
    "admin_token": User(id="1", username="admin", roles=[UserRole.ADMIN]),
    "cra_token": User(id="2", username="cra_jane", roles=[UserRole.CRA]),
    "dm_token": User(id="3", username="dm_bob", roles=[UserRole.DATA_MANAGER]),
    "exec_token": User(id="4", username="exec_sarah", roles=[UserRole.EXECUTIVE]),
}

rbac_service = RBACService()

async def get_current_user(x_token: Optional[str] = Header(None)) -> User:
    """
    Simulates authentication by checking X-Token header.
    In production, this would verify a JWT token.
    """
    if not x_token:
        # For development ease, default to Admin if no token provided (simulating dev mode)
        # In PROD this should raise 401
        return User(id="1", username="dev_admin", roles=[UserRole.ADMIN])
        
    if x_token in MOCK_USERS:
        return MOCK_USERS[x_token]
        
    raise HTTPException(status_code=401, detail="Invalid Authentication Token")

def require_role(role: UserRole):
    """
    Dependency to enforce role presence.
    """
    def _role_checker(user: User = Depends(get_current_user)):
        if role not in user.roles and UserRole.ADMIN not in user.roles:
            raise HTTPException(status_code=403, detail=f"Role {role} required")
        return user
    return _role_checker

def require_permission(permission: str):
    """
    Dependency to enforce fine-grained permission.
    """
    def _perm_checker(user: User = Depends(get_current_user)):
        if not rbac_service.has_permission(user, permission):
            raise HTTPException(status_code=403, detail=f"Permission {permission} required")
        return user
    return _perm_checker
