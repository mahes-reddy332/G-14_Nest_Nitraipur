
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import uuid
import logging

class AuditEventType(str):
    DATA_INGESTION = "DATA_INGESTION"
    RISK_SCORING = "RISK_SCORING"
    AI_GENERATION = "AI_GENERATION"
    USER_ACTION = "USER_ACTION"
    SYSTEM_ERROR = "SYSTEM_ERROR"

class AuditLog(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str
    user_id: str  # Who?
    action: str   # What?
    details: Dict[str, Any] = {} # Why/Context?
    status: str = "SUCCESS" 
    client_ip: Optional[str] = None

class AuditService:
    """
    Central service for structured audit logging.
    Ensures traceability of all critical system actions.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("AuditLogger")
        # In a real system, this would be a database connection (Postgres/Mongo)
        # or a dedicated audit system (Splunk/Elastic)
        self._memory_store: List[AuditLog] = [] 

    async def log_event(self, 
                  event_type: str, 
                  user_id: str, 
                  action: str, 
                  details: Dict[str, Any] = None,
                  status: str = "SUCCESS",
                  client_ip: str = None) -> AuditLog:
        
        log_entry = AuditLog(
            event_type=event_type,
            user_id=user_id,
            action=action,
            details=details or {},
            status=status,
            client_ip=client_ip
        )
        
        # Persist log
        self._memory_store.append(log_entry)
        
        # Also emit to standard logger for container capture
        self.logger.info(f"AUDIT_EVENT: {log_entry.json()}")
        
        return log_entry

    async def get_logs(self, 
                 user_id: Optional[str] = None, 
                 event_type: Optional[str] = None,
                 limit: int = 100) -> List[AuditLog]:
        """
        Retrieve logs with basic filtering.
        """
        logs = self._memory_store
        
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
            
        if event_type:
            logs = [l for l in logs if l.event_type == event_type]
            
        # Sort by latest first
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        return logs[:limit]
