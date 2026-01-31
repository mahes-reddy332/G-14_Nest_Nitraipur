
from enum import Enum
from typing import List, Dict, Set
from pydantic import BaseModel
import logging

class UserRole(str, Enum):
    DATA_MANAGER = "data_manager"
    CRA = "cra" # Clinical Research Associate
    SAFETY_REVIEWER = "safety_reviewer"
    STUDY_LEAD = "study_lead"
    EXECUTIVE = "executive"
    ADMIN = "admin"

class Permission(str, Enum):
    VIEW_DASHBOARD = "view:dashboard"
    VIEW_PHI = "view:phi" # Protected Health Information
    EDIT_DATA = "edit:data"
    APPROVE_RISK = "approve:risk"
    GENERATE_AI = "generate:ai"
    EXPORT_AUDIT = "export:audit"
    MANAGE_USERS = "manage:users"

class User(BaseModel):
    id: str
    username: str
    roles: List[UserRole]
    
    @property
    def is_admin(self) -> bool:
        return UserRole.ADMIN in self.roles

class RBACService:
    """
    Manages Role-Based Access Control policies.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Define Policy Matrix
        self.role_permissions: Dict[UserRole, Set[Permission]] = {
            UserRole.ADMIN: set(Permission), # All permissions
            
            UserRole.STUDY_LEAD: {
                Permission.VIEW_DASHBOARD,
                Permission.VIEW_PHI,
                Permission.EDIT_DATA,
                Permission.APPROVE_RISK,
                Permission.GENERATE_AI,
                Permission.EXPORT_AUDIT
            },
            
            UserRole.DATA_MANAGER: {
                Permission.VIEW_DASHBOARD,
                Permission.VIEW_PHI,
                Permission.EDIT_DATA,
                Permission.GENERATE_AI
            },
            
            UserRole.CRA: {
                Permission.VIEW_DASHBOARD,
                Permission.EDIT_DATA,
                Permission.APPROVE_RISK
            },
            
            UserRole.SAFETY_REVIEWER: {
                Permission.VIEW_DASHBOARD,
                Permission.VIEW_PHI,
                Permission.APPROVE_RISK
            },
            
            UserRole.EXECUTIVE: {
                Permission.VIEW_DASHBOARD,
                Permission.EXPORT_AUDIT
            }
        }

    def has_permission(self, user: User, required_permission: Permission) -> bool:
        """
        Check if user has specific permission through any of their roles.
        """
        for role in user.roles:
            if role not in self.role_permissions:
                continue
            if required_permission in self.role_permissions[role]:
                return True
        
        self.logger.warning(f"Access Denied: User {user.username} missing {required_permission}")
        return False
        
    def get_user_permissions(self, user: User) -> List[str]:
        """
        Return all permissions for a user.
        """
        perms = set()
        for role in user.roles:
            if role in self.role_permissions:
                perms.update(self.role_permissions[role])
        return list(perms)
