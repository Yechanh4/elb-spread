"""Daily refresh with APScheduler. The FastAPI app (step 3) starts this in-process; standalone:
    .venv/bin/python -m data.cli run-scheduler
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

JOB_ID = "daily-refresh"
DOW = {0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri", 5: "sat", 6: "sun"}


def trigger_for(schedule):
    return CronTrigger(day_of_week=",".join(DOW[d] for d in schedule.weekdays), hour=schedule.hour,
                       minute=schedule.minute, timezone=schedule.tz)


def build_scheduler(service, scheduler=None):
    sch = scheduler or BackgroundScheduler()
    sch.add_job(service.refresh_all, trigger_for(service.schedule), id=JOB_ID, replace_existing=True,
                coalesce=True, max_instances=1, misfire_grace_time=3600)
    return sch


def reschedule(sch, service, hour, minute, tz="America/New_York"):
    service.set_schedule(hour, minute, tz)
    sch.reschedule_job(JOB_ID, trigger=trigger_for(service.schedule))
    return sch.get_job(JOB_ID)
