"""Evidence-provider status formatting for the /api/health endpoint.

Extracted out of app/main.py so it can be shared with app/main_beta.py (the
beta build's entry point) without app/main_beta.py needing to import
app/main.py itself -- which would pull in every instructor-only router
(authoring, references, submissions, analytics, mentor_review,
mentor_insights, settings) that the beta build deliberately excludes.
No behavior change from the original app/main.py::_evidence_status.
"""
import sys

from app.services.native_mac import NativeMacEvidenceProvider
from app.services.native_rust import RustCaptureEvidenceProvider
from app.services.native_windows import NativeWindowsEvidenceProvider


async def get_evidence_status(provider) -> dict:
    """Provider-aware status block. native_mac reports permission-level
    detail instead of a URL, since that's what actually determines
    readiness for it."""
    if isinstance(provider, NativeMacEvidenceProvider):
        connected = await provider.health()
        permissions = await provider.permission_status() if connected else {
            "screen_recording": False, "accessibility": False, "input_monitoring": False,
        }
        all_granted = all(permissions.values())
        if not connected:
            hint = f"Native capture executable not found or not runnable at {provider.executable}"
        elif not all_granted:
            missing = [name.replace("_", " ").title() for name, ok in permissions.items() if not ok]
            hint = f"Missing macOS permission(s): {', '.join(missing)}. Grant them in System Settings."
        else:
            hint = None
        return {
            "label": "Native macOS",
            "connected": connected,
            "evidence_access": connected and all_granted,
            "permissions": permissions,
            "hint": hint,
        }

    if isinstance(provider, NativeWindowsEvidenceProvider):
        connected = await provider.health()
        permissions = await provider.permission_status() if connected else {
            "screen_capture": False, "ui_automation": False, "foreground_window": False,
        }
        all_granted = all(permissions.values())
        if not connected:
            hint = f"Native capture executable not found or not runnable at {provider.executable}"
        elif not all_granted:
            missing = [name.replace("_", " ").title() for name, ok in permissions.items() if not ok]
            hint = (
                f"Missing Windows capability/capture permission(s): {', '.join(missing)}. "
                "Run `mentor-capture.exe status` for details."
            )
        else:
            hint = None
        return {
            "label": "Native Windows",
            "connected": connected,
            "evidence_access": connected and all_granted,
            "permissions": permissions,
            "hint": hint,
        }

    if isinstance(provider, RustCaptureEvidenceProvider):
        connected = await provider.health()
        # The Rust helper reports different permissions per platform: macOS
        # needs Screen Recording + Accessibility + Input Monitoring (TCC);
        # Windows reports functional capabilities (Screen Capture, UI Automation,
        # Input Hooks, Active Window). Use the right fallback shape so the
        # status bar wording matches the platform the user is actually on.
        if sys.platform == "darwin":
            fallback = {
                "screen_recording": False, "accessibility": False, "input_monitoring": False,
            }
            missing_label = "macOS permission(s)"
            missing_hint = "Grant them in System Settings."
        else:
            fallback = {
                "screen_capture": False, "ui_automation": False,
                "input_hooks": False, "active_window": False,
            }
            missing_label = "Windows capability/capture permission(s)"
            missing_hint = "Run `cyberalfred-capture.exe check` for details."
        permissions = await provider.permission_status() if connected else fallback
        all_granted = all(permissions.values())
        if not connected:
            hint = f"Native capture executable not found or not runnable at {provider.executable}"
        elif not all_granted:
            missing = [name.replace("_", " ").title() for name, ok in permissions.items() if not ok]
            hint = f"Missing {missing_label}: {', '.join(missing)}. {missing_hint}"
        else:
            hint = None
        return {
            "label": "Native Rust",
            "connected": connected,
            "evidence_access": connected and all_granted,
            "permissions": permissions,
            "hint": hint,
        }

    # Unknown provider type -- degrade gracefully rather than crash the page.
    connected = await provider.health()
    return {"label": type(provider).__name__, "connected": connected, "evidence_access": connected, "hint": None}
