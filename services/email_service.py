"""Outgoing email over SMTP (Gmail by default).

Used to deliver password-reset codes. Credentials come from the environment -
for Gmail, ``SMTP_PASSWORD`` must be a 16-character **App Password** created at
https://myaccount.google.com/apppasswords (Google blocks plain account
passwords for SMTP). Nothing is ever hardcoded here.
"""

import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional

from config import settings


def is_configured() -> bool:
    """True when an SMTP account is set up, so mail can actually be sent."""
    return settings.email_is_configured()


def send_email(to: str, subject: str, body: str, html: Optional[str] = None) -> dict:
    """Send one email. Returns ``{"success": bool, "error": str | None}``.

    Never raises: a mail failure must not take down the page that called it.
    """
    if not is_configured():
        return {"success": False, "error": "Email is not configured on this server."}

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_USER}>"
    message["To"] = to
    message.set_content(body)
    if html:
        message.add_alternative(html, subtype="html")

    try:
        context = ssl.create_default_context()
        if settings.SMTP_PORT == 465:
            with smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT,
                                  context=context, timeout=20) as server:
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.send_message(message)
        else:
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20) as server:
                server.ehlo()
                server.starttls(context=context)
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.send_message(message)
        return {"success": True, "error": None}
    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "error": "SMTP login was rejected. For Gmail, use a 16-character "
                     "App Password (not your normal password) and make sure "
                     "2-Step Verification is on.",
        }
    except Exception as exc:  # network, DNS, TLS, timeout...
        return {"success": False, "error": f"Could not send the email: {exc}"}


def send_reset_code(to: str, name: str, code: str, minutes: int) -> dict:
    """Send a password-reset code, formatted as plain text and HTML."""
    app = settings.APP_NAME
    body = (
        f"Hi {name},\n\n"
        f"Your {app} password reset code is: {code}\n\n"
        f"It expires in {minutes} minutes and can only be used once.\n"
        f"If you did not ask to reset your password, you can ignore this email "
        f"- your current password still works.\n\n"
        f"- The {app} team"
    )
    html = f"""
    <div style="font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:480px">
      <h2 style="margin-bottom:4px">{app}</h2>
      <p>Hi {name},</p>
      <p>Use this code to reset your password:</p>
      <p style="font-size:30px;font-weight:700;letter-spacing:6px;
                background:#f4f4f5;padding:14px 18px;border-radius:8px;
                display:inline-block">{code}</p>
      <p style="color:#555">It expires in {minutes} minutes and can only be used once.</p>
      <p style="color:#555">If you did not ask to reset your password, you can
         ignore this email - your current password still works.</p>
    </div>
    """
    return send_email(to, f"{app} password reset code", body, html)


if __name__ == "__main__":
    # Quick check that your SMTP settings work (run from the project root):
    #     python -m services.email_service you@gmail.com
    import sys

    if not is_configured():
        print("SMTP is not configured. Set SMTP_USER and SMTP_PASSWORD in .env.")
        raise SystemExit(1)
    recipient = sys.argv[1] if len(sys.argv) > 1 else settings.SMTP_USER
    print(f"Sending a test email from {settings.SMTP_USER} to {recipient}...")
    outcome = send_email(
        recipient,
        f"{settings.APP_NAME} test email",
        "If you can read this, SmartBite AI can send password-reset codes.",
    )
    print("Sent." if outcome["success"] else f"Failed: {outcome['error']}")
    raise SystemExit(0 if outcome["success"] else 1)
