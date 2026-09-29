"""Email delivery for facilitator escalations."""

from email.message import EmailMessage
import os
import smtplib


def send_escalation_email(*, user_id, timestamp, jurisdiction, question,
                          assistant_response, reason, contact_email=""):
    """Return delivery status and a safe error string without leaking SMTP details."""
    names = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "FACILITATOR_EMAIL")
    config = {name: os.getenv(name) for name in names}
    if any(not value for value in config.values()):
        return "not_configured", None

    message = EmailMessage()
    message["Subject"] = "IP-SAKTI Sahayak facilitator escalation"
    message["From"] = config["SMTP_USER"]
    message["To"] = config["FACILITATOR_EMAIL"]
    if contact_email.strip():
        message["Reply-To"] = contact_email.strip()
    message.set_content(
        f"User ID: {user_id}\n"
        f"Timestamp: {timestamp}\n"
        f"Jurisdiction: {jurisdiction}\n\n"
        f"Original query:\n{question}\n\n"
        f"Assistant's last response:\n{assistant_response}\n\n"
        f"Reason for escalation:\n{reason}\n"
    )
    try:
        with smtplib.SMTP(config["SMTP_HOST"], int(config["SMTP_PORT"]), timeout=10) as server:
            server.starttls()
            server.login(config["SMTP_USER"], config["SMTP_PASS"])
            server.send_message(message)
    except Exception:
        return "failed", "Email delivery failed. Check the SMTP configuration and try again."
    return "sent", None
