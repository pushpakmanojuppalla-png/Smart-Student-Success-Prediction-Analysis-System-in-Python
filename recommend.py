"""Rule-based, explainable recommendations for an individual student."""
import math


def _get(student: dict, key: str, default=None):
    value = student.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return value


def generate_recommendations(student: dict, level: str) -> list:
    recs = []

    if level == "At-Risk":
        recs.append("Enrol in the early-intervention programme: assign a mentor and "
                    "schedule weekly progress check-ins.")

    attendance = _get(student, "attendance_pct")
    if attendance is not None and attendance < 75:
        recs.append(f"Attendance is {attendance:.0f}%. Aim for at least 85% by "
                    "setting reminders and discussing barriers with an advisor.")

    study = _get(student, "study_hours_per_week")
    if study is not None and study < 10:
        recs.append(f"Only {study:.0f} study hours/week. Build a timetable with "
                    "12-15 focused hours and use short Pomodoro sessions.")

    assignment = _get(student, "assignment_completion_pct")
    if assignment is not None and assignment < 70:
        recs.append("Assignment completion is low. Break work into weekly "
                    "milestones and request extension support early.")

    gpa = _get(student, "previous_gpa")
    if gpa is not None and gpa < 2.5:
        recs.append("Previous GPA is low. Join tutoring or remedial sessions for "
                    "the weakest subjects.")

    participation = _get(student, "participation_score")
    if participation is not None and participation < 4:
        recs.append("Class participation is low. Try study groups and ask at "
                    "least one question per class.")

    stress = _get(student, "stress_level")
    if stress is not None and stress >= 8:
        recs.append("Stress level is high. Offer counselling / wellness "
                    "resources and time-management coaching.")

    sleep = _get(student, "sleep_hours")
    if sleep is not None and (sleep < 6 or sleep > 9):
        recs.append(f"Sleep ({sleep:.1f} h) is outside the healthy 6-9 h range. "
                    "Encourage a consistent sleep schedule.")

    if _get(student, "internet_access") == "No":
        recs.append("No home internet. Provide access to campus labs / library "
                    "hours or offline learning material.")

    if _get(student, "family_support") == "Low":
        recs.append("Low family support. Connect the student with a counsellor "
                    "and peer-mentor network.")

    absences = _get(student, "absences")
    if absences is not None and absences >= 10:
        recs.append(f"{int(absences)} absences recorded. Review reasons and "
                    "agree on a catch-up plan.")

    if not recs:
        recs.append("On track. Keep up current habits and consider peer-tutoring "
                    "or advanced enrichment activities.")
    return recs
