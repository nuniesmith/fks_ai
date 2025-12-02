"""
Timezone utilities for FKS AI service.
Provides EST/EDT aware datetime functions.
"""
from datetime import datetime
import pytz

# Eastern timezone (handles EST/EDT automatically)
EASTERN = pytz.timezone('America/New_York')


def now_est() -> datetime:
    """
    Get current datetime in Eastern timezone (EST/EDT).
    
    Returns:
        datetime: Current time in Eastern timezone
        
    Examples:
        >>> now = now_est()
        >>> now.tzinfo
        <DstTzInfo 'America/New_York' EST-1 day, 19:00:00 STD>
    """
    return datetime.now(EASTERN)


def now_est_iso() -> str:
    """
    Get current datetime in Eastern timezone as ISO format string.
    
    Returns:
        str: ISO format datetime string with timezone
        
    Examples:
        >>> now_est_iso()
        '2025-11-28T18:30:00-05:00'  # EST
        '2025-06-28T18:30:00-04:00'  # EDT
    """
    return now_est().isoformat()


def utc_to_est(utc_dt: datetime) -> datetime:
    """
    Convert UTC datetime to Eastern timezone.
    
    Args:
        utc_dt: UTC datetime (naive or timezone-aware)
        
    Returns:
        datetime: Eastern timezone datetime
        
    Examples:
        >>> from datetime import datetime
        >>> utc_dt = datetime(2025, 11, 28, 23, 30, 0)
        >>> est_dt = utc_to_est(utc_dt)
    """
    # If naive, assume UTC
    if utc_dt.tzinfo is None:
        utc_dt = pytz.UTC.localize(utc_dt)
    
    # Convert to Eastern
    return utc_dt.astimezone(EASTERN)
