"""Daily trading sessions for the exchanges listed by the data sources."""

import exchange_calendars as xcals
import pandas as pd


def calendar_and_sessions(name, start, end):
    """Return the exchange calendar and session labels between two inclusive dates."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if any(pd.isna(day) or day.tz is not None or day != day.normalize()
           for day in (start, end)) or start > end:
        raise ValueError("Enter local dates without times in ascending order.")
    margin = pd.Timedelta(days=15)
    calendar = xcals.get_calendar(name, start=start - margin, end=end + margin)
    sessions = calendar.sessions_in_range(start, end)
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    return calendar, sessions


def expected_sessions(name, start, end):
    """Expected sessions require a known exchange calendar."""
    if name is None:
        return None
    _, sessions = calendar_and_sessions(name, start, end)
    return sessions
