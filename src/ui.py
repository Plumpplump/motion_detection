"""Flet dashboard for motion-triggered object recognition.

The layout is deliberately calm when idle and loud only when the watched
object is actually found.  All backend-specific logic lives in ``vision/`` so
this module only deals with presentation and user input.
"""

from __future__ import annotations

import asyncio
import time

import flet as ft

from vision.catalog import (
    COCO_OBJECTS,
    CUSTOM_OBJECT_KEY,
    DEFAULT_OBJECT_KEY,
    ObjectSpec,
    find_object,
    make_custom_target,
)
from vision.providers import PROVIDERS, BackendConfig
from vision.types import DetectionResult

# Palette
BACKGROUND = "#F3F5F9"
SURFACE = "#FFFFFF"
INK = "#0F172A"
MUTED = "#64748B"
SUBTLE = "#94A3B8"
BORDER = "#E2E8F0"
ACCENT = "#4F46E5"
ACCENT_SOFT = "#EEF2FF"
GREEN = "#059669"
GREEN_SOFT = "#ECFDF5"
RED = "#DC2626"
RED_DARK = "#B91C1C"
RED_SOFT = "#FEF2F2"
AMBER = "#D97706"
AMBER_SOFT = "#FFFBEB"

INTERVAL_MIN = 0.5
INTERVAL_MAX = 3.0


def _field_kwargs() -> dict:
    """Shared input styling that works on Flet 0.86 and 1.0+."""

    kwargs: dict = {"filled": True, "bgcolor": "#F8FAFC"}
    if hasattr(ft, "OutlineInputBorder"):
        kwargs["border"] = {
            ft.ControlState.DEFAULT: ft.OutlineInputBorder(
                border_radius=12,
                side=ft.BorderSide(1, BORDER),
            ),
            ft.ControlState.FOCUSED: ft.OutlineInputBorder(
                border_radius=12,
                side=ft.BorderSide(2, ACCENT),
            ),
        }
    else:
        kwargs.update(
            border_radius=12,
            border_color=BORDER,
            focused_border_color=ACCENT,
        )
    return kwargs


FIELD_KWARGS = _field_kwargs()





def _section_header(
    title: str,
    icon: ft.IconData,
    subtitle: str | None = None,
) -> ft.Control:
    text_column = [
        ft.Text(title, size=14, weight=ft.FontWeight.W_700, color=INK),
    ]
    if subtitle:
        text_column.append(
            ft.Text(subtitle, size=11, color=MUTED, max_lines=2),
        )
    return ft.Row(
        [
            ft.Container(
                content=ft.Icon(icon, size=18, color=ACCENT),
                bgcolor=ACCENT_SOFT,
                padding=8,
                border_radius=10,
            ),
            ft.Column(text_column, spacing=2, expand=True),
        ],
        spacing=10,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


class Dashboard:
    def __init__(
        self,
        page: ft.Page,
        *,
        on_toggle_camera,
        on_toggle_detection,
        on_apply_settings,
        on_dismiss_warning,
        on_target_change=None,
    ) -> None:
        self.page = page
        self.on_toggle_camera = on_toggle_camera
        self.on_toggle_detection = on_toggle_detection
        self.on_apply_settings = on_apply_settings
        self.on_dismiss_warning = on_dismiss_warning
        self.on_target_change = on_target_change
        self._build_controls()

    # ------------------------------------------------------------------ build
    def _build_controls(self) -> None:
        self.status_dot = ft.Container(
            width=8,
            height=8,
            border_radius=99,
            bgcolor=SUBTLE,
        )
        self.status = ft.Text(
            "Camera off",
            key="status",
            size=13,
            color=MUTED,
            max_lines=3,
            expand=True,
        )
        self.check_spinner = ft.ProgressRing(
            width=14,
            height=14,
            stroke_width=2,
            color=ACCENT,
            visible=False,
        )
        self.status_chip = ft.Container(
            bgcolor="#F8FAFC",
            border_radius=99,
            padding=ft.Padding(left=12, top=8, right=12, bottom=8),
            content=ft.Row(
                [self.status_dot, self.status, self.check_spinner],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

        self.preview_img = ft.RawImage(fit=ft.BoxFit.CONTAIN)
        self.camera_surface = ft.Container(
            content=self.preview_img,
            alignment=ft.Alignment.CENTER,
        )
        self.camera_placeholder = ft.Container(
            alignment=ft.Alignment.CENTER,
            content=ft.Column(
                [
                    ft.Icon(ft.Icons.VIDEOCAM_OFF, size=42, color=SUBTLE),
                    ft.Text(
                        "Waiting for camera",
                        size=14,
                        weight=ft.FontWeight.W_600,
                        color="#E2E8F0",
                    ),
                    ft.Text(
                        "Start the camera to begin watching",
                        size=11,
                        color=SUBTLE,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=6,
            ),
        )
        self.live_chip = ft.Container(
            left=14,
            top=14,
            visible=False,
            bgcolor=GREEN,
            padding=ft.Padding(left=9, top=5, right=10, bottom=5),
            border_radius=99,
            content=ft.Row(
                [
                    ft.Container(
                        width=7,
                        height=7,
                        border_radius=99,
                        bgcolor=ft.Colors.WHITE,
                    ),
                    ft.Text(
                        "LIVE",
                        size=9,
                        weight=ft.FontWeight.W_700,
                        color=ft.Colors.WHITE,
                    ),
                ],
                spacing=5,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )
        self.camera_flash = ft.Container(
            bgcolor=ft.Colors.with_opacity(0.42, RED),
            visible=False,
        )
        self.camera_box = ft.Container(
            key="camera_box",
            height=420,
            bgcolor="#0B1220",
            border_radius=18,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            content=ft.Stack(
                [
                    self.camera_surface,
                    self.camera_flash,
                    self.camera_placeholder,
                    self.live_chip,
                ],
                fit=ft.StackFit.EXPAND,
            ),
        )

        self.warning_title = ft.Text(
            "OBJECT DETECTED",
            size=16,
            weight=ft.FontWeight.W_800,
            color=ft.Colors.WHITE,
        )
        self.warning_sub = ft.Text(
            "",
            size=11,
            color=ft.Colors.WHITE_70,
            max_lines=2,
        )
        self.warning_thumb = ft.Image(
            src=b"",
            width=56,
            height=56,
            fit=ft.BoxFit.COVER,
            border_radius=10,
            visible=False,
        )
        self.warning_card = ft.Container(
            key="warning_banner",
            visible=False,
            border_radius=18,
            padding=16,
            gradient=ft.LinearGradient(
                begin=ft.Alignment.TOP_LEFT,
                end=ft.Alignment.BOTTOM_RIGHT,
                colors=[RED, RED_DARK],
            ),
            shadow=ft.BoxShadow(
                blur_radius=24,
                color=ft.Colors.with_opacity(0.30, RED),
                offset=(0, 8),
            ),
            content=ft.Row(
                [
                    ft.Container(
                        content=ft.Icon(
                            ft.Icons.WARNING_AMBER_ROUNDED,
                            size=26,
                            color=ft.Colors.WHITE,
                        ),
                        bgcolor=ft.Colors.with_opacity(0.18, ft.Colors.WHITE),
                        padding=10,
                        border_radius=12,
                    ),
                    ft.Column(
                        [self.warning_title, self.warning_sub],
                        spacing=2,
                        expand=True,
                    ),
                    self.warning_thumb,
                    ft.IconButton(
                        icon=ft.Icons.CLOSE,
                        icon_color=ft.Colors.WHITE,
                        tooltip="Dismiss",
                        on_click=self.on_dismiss_warning,
                    ),
                ],
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

        button_shape = ft.RoundedRectangleBorder(radius=12)
        self.start_camera_btn = ft.FilledButton(
            content="Start Camera",
            key="start_camera",
            icon=ft.Icons.VIDEOCAM,
            on_click=self.on_toggle_camera,
            bgcolor=ACCENT,
            color=ft.Colors.WHITE,
            style=ft.ButtonStyle(
                shape=button_shape,
                padding=ft.Padding(left=16, top=14, right=16, bottom=14),
            ),
        )
        self.start_detection_btn = ft.FilledButton(
            content="Start Detection",
            key="start_detection",
            icon=ft.Icons.MOTION_PHOTOS_ON,
            disabled=True,
            on_click=self.on_toggle_detection,
            bgcolor=INK,
            color=ft.Colors.WHITE,
            style=ft.ButtonStyle(
                shape=button_shape,
                padding=ft.Padding(left=16, top=14, right=16, bottom=14),
            ),
        )

        self.ai_text = ft.Text(
            "Local AI starting…",
            key="ai_status",
            size=11,
            weight=ft.FontWeight.W_600,
            color=ACCENT,
            max_lines=3,
            expand=True,
        )
        self.ai_icon = ft.Icon(ft.Icons.MEMORY, size=14, color=ACCENT)
        self.ai_chip = ft.Container(
            bgcolor=ACCENT_SOFT,
            border_radius=99,
            padding=ft.Padding(left=10, top=6, right=10, bottom=6),
            content=ft.Row(
                [self.ai_icon, self.ai_text],
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

        self.object_dropdown = ft.Dropdown(
            key="object_select",
            value=DEFAULT_OBJECT_KEY,
            options=[
                ft.DropdownOption(key=obj.key, text=obj.display)
                for obj in COCO_OBJECTS
            ]
            + [ft.DropdownOption(key=CUSTOM_OBJECT_KEY, text="🎯  Custom object…")],
            on_select=self._on_object_select,
            **FIELD_KWARGS,
        )
        self.custom_field = ft.TextField(
            key="custom_object",
            label="Custom object",
            hint_text="e.g. wallet, keys, red mug",
            visible=False,
            on_change=self._on_custom_change,
            **FIELD_KWARGS,
        )
        self.custom_hint = ft.Text(
            "Custom objects need OpenRouter or another remote backend.",
            size=10,
            color=AMBER,
            visible=False,
        )
        self.object_hint = ft.Text(
            "Local label: cell phone",
            size=10,
            color=SUBTLE,
        )

        self.interval_slider = ft.Slider(
            key="interval_slider",
            min=INTERVAL_MIN,
            max=INTERVAL_MAX,
            value=1.0,
            divisions=10,
            label="1.0 s",
            active_color=ACCENT,
            on_change=self._on_interval_change,
        )
        self.interval_value = ft.Text(
            "1.0 s",
            size=12,
            weight=ft.FontWeight.W_700,
            color=ACCENT,
        )

        self.provider_dropdown = ft.Dropdown(
            key="provider_select",
            value="embedded",
            options=[
                ft.DropdownOption(key=preset.key, text=f"{preset.emoji}  {preset.label}")
                for preset in PROVIDERS.values()
            ],
            on_select=self._on_provider_select,
            **FIELD_KWARGS,
        )
        self.provider_note = ft.Text(
            PROVIDERS["embedded"].description,
            size=10,
            color=MUTED,
            max_lines=2,
        )
        self.local_model_chip = ft.Container(
            bgcolor=GREEN_SOFT,
            border_radius=10,
            padding=ft.Padding(left=10, top=8, right=10, bottom=8),
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.MEMORY, size=14, color=GREEN),
                    ft.Text(
                        "YOLOX-Tiny INT8 · 5.3 MB · on-device",
                        size=10,
                        weight=ft.FontWeight.W_600,
                        color=GREEN,
                    ),
                ],
                spacing=6,
            ),
        )
        self.model_field = ft.TextField(
            key="model_field",
            label="Model",
            visible=False,
            **FIELD_KWARGS,
        )
        self.base_url_field = ft.TextField(
            key="base_url_field",
            label="Base URL",
            visible=False,
            **FIELD_KWARGS,
        )
        self.api_key_field = ft.TextField(
            key="api_key_field",
            label="API key",
            password=True,
            can_reveal_password=True,
            visible=False,
            **FIELD_KWARGS,
        )
        self.key_hint = ft.Text(
            "Leave blank to use OPENROUTER_API_KEY",
            size=10,
            color=SUBTLE,
            visible=False,
        )
        self.threshold_slider = ft.Slider(
            key="threshold_slider",
            min=0.1,
            max=0.9,
            value=0.35,
            divisions=16,
            label="35%",
            active_color=ACCENT,
            on_change=self._on_threshold_change,
        )
        self.threshold_label = ft.Text(
            "Confidence threshold: 35%",
            size=11,
            color=MUTED,
        )
        self.alert_switch = ft.Switch(
            key="alert_switch",
            value=True,
            label="Flash and sound when the object is found",
            label_position=ft.LabelPosition.RIGHT,
            active_color=RED,
        )
        self.apply_btn = ft.FilledButton(
            content="Apply backend",
            key="apply_settings",
            icon=ft.Icons.CHECK_CIRCLE,
            on_click=self.on_apply_settings,
            bgcolor=INK,
            color=ft.Colors.WHITE,
            style=ft.ButtonStyle(
                shape=button_shape,
                padding=ft.Padding(left=14, top=12, right=14, bottom=12),
            ),
        )

        # ------------------------------------------------------------------ layout
    def build(self) -> ft.Control:
        header = ft.Row(
            [
                ft.Container(
                    content=ft.Icon(ft.Icons.SECURITY, size=28, color=ACCENT),
                    bgcolor=ACCENT_SOFT,
                    padding=12,
                    border_radius=14,
                ),
                ft.Column(
                    [
                        ft.Text(
                            "Motion Sentinel",
                            size=26,
                            weight=ft.FontWeight.W_800,
                            color=INK,
                        ),
                        ft.Text(
                            "Motion-triggered object watch",
                            size=12,
                            color=MUTED,
                        ),
                    ],
                    spacing=0,
                    expand=True,
                ),
                ft.Container(
                    content=ft.Text(
                        "LOCAL-FIRST",
                        size=9,
                        weight=ft.FontWeight.W_800,
                        color=GREEN,
                    ),
                    bgcolor=GREEN_SOFT,
                    padding=ft.Padding(left=8, top=4, right=8, bottom=4),
                    border_radius=99,
                ),
            ],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        controls_row = ft.Row(
            [
                self.start_camera_btn,
                self.start_detection_btn,
            ],
            spacing=10,
            wrap=True,
            alignment=ft.MainAxisAlignment.CENTER,
        )

        status_row = ft.Column(
            [self.status_chip, self.ai_chip],
            spacing=8,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        footer = ft.Row(
            [
                ft.Icon(ft.Icons.SHIELD_OUTLINED, size=14, color=SUBTLE),
                ft.Text(
                    "Local-first. Frames stay on this computer unless you select a remote backend.",
                    size=10,
                    color=SUBTLE,
                    expand=True,
                ),
            ],
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        # ------------------------------------------------------------ sidebar
        sidebar = ft.Container(
            col={"xs": 12, "md": 4, "lg": 4, "xl": 3},
            bgcolor=SURFACE,
            border_radius=20,
            padding=20,
            border=ft.Border.all(1, BORDER),
            shadow=ft.BoxShadow(
                blur_radius=20,
                color=ft.Colors.with_opacity(0.07, INK),
                offset=(0, 6),
            ),
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Container(
                                content=ft.Icon(
                                    ft.Icons.TUNE,
                                    size=18,
                                    color=ACCENT,
                                ),
                                bgcolor=ACCENT_SOFT,
                                padding=8,
                                border_radius=10,
                            ),
                            ft.Column(
                                [
                                    ft.Text(
                                        "Settings",
                                        size=16,
                                        weight=ft.FontWeight.W_700,
                                        color=INK,
                                    ),
                                    ft.Text(
                                        "Tune what gets watched",
                                        size=10,
                                        color=MUTED,
                                    ),
                                ],
                                spacing=0,
                                expand=True,
                            ),
                        ],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Divider(height=1, color=BORDER),
                    _section_header(
                        "Object to watch",
                        ft.Icons.CENTER_FOCUS_STRONG,
                    ),
                    self.object_dropdown,
                    self.custom_field,
                    self.custom_hint,
                    self.object_hint,
                    ft.Divider(height=1, color=BORDER),
                    _section_header(
                        "Motion sensitivity",
                        ft.Icons.MOTION_PHOTOS_ON,
                    ),
                    ft.Row(
                        [
                            ft.Text(
                                "Compare to frame from",
                                size=11,
                                color=MUTED,
                            ),
                            ft.Container(expand=True),
                            self.interval_value,
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self.interval_slider,
                    ft.Divider(height=1, color=BORDER),
                    _section_header(
                        "AI backend",
                        ft.Icons.MEMORY,
                    ),
                    self.provider_dropdown,
                    self.provider_note,
                    self.local_model_chip,
                    self.model_field,
                    self.base_url_field,
                    self.api_key_field,
                    self.key_hint,
                    self.threshold_label,
                    self.threshold_slider,
                    self.apply_btn,
                    ft.Divider(height=1, color=BORDER),
                    self.alert_switch,
                    ft.Row(
                        [
                            ft.Icon(
                                ft.Icons.LOCK_OUTLINE,
                                size=13,
                                color=SUBTLE,
                            ),
                            ft.Text(
                                "Frames stay on this computer with the local model.",
                                size=10,
                                color=SUBTLE,
                                expand=True,
                            ),
                        ],
                        spacing=6,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ],
                spacing=12,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            ),
        )

        # --------------------------------------------------------------- main
        main = ft.Container(
            col={"xs": 12, "md": 8, "lg": 8, "xl": 9},
            content=ft.Column(
                [
                    header,
                    self.warning_card,
                    ft.Container(
                        content=self.camera_box,
                        border_radius=18,
                        shadow=ft.BoxShadow(
                            blur_radius=26,
                            color=ft.Colors.with_opacity(0.16, "#0B1220"),
                            offset=(0, 12),
                        ),
                    ),
                    controls_row,
                    status_row,
                    footer,
                ],
                spacing=16,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
        )

        return ft.SafeArea(
            expand=True,
            content=ft.Container(
                expand=True,
                padding=ft.Padding(left=22, top=22, right=22, bottom=22),
                content=ft.ResponsiveRow(
                    [sidebar, main],
                    columns=12,
                    spacing=20,
                    run_spacing=20,
                    expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
            ),
        )

    # ----------------------------------------------------------------- actions
    def _on_object_select(self, _event=None) -> None:
        custom = self.object_dropdown.value == CUSTOM_OBJECT_KEY
        self.custom_field.visible = custom
        self.custom_hint.visible = custom
        target = self.get_target()
        if target is None:
            self.object_hint.value = "Type an object name to watch."
        elif target.supported_locally:
            self.object_hint.value = f"Local label: {target.model_label}"
        else:
            self.object_hint.value = (
                "Custom objects need OpenRouter or a local VLM server."
            )
        self.page.update()
        if self.on_target_change is not None:
            self.on_target_change(target)

    def _on_custom_change(self, _event=None) -> None:
        self._on_object_select()

    def _on_provider_select(self, _event=None) -> None:
        provider = self.provider_dropdown.value or "embedded"
        preset = PROVIDERS[provider]
        is_embedded = provider == "embedded"
        is_ollama = provider == "ollama"
        is_custom = provider == "custom"

        self.provider_note.value = preset.description
        self.model_field.visible = not is_embedded
        self.base_url_field.visible = is_ollama or is_custom
        self.api_key_field.visible = provider in {"openrouter", "custom"}
        self.key_hint.visible = provider in {"openrouter", "custom"}
        self.local_model_chip.visible = is_embedded
        self.model_field.value = preset.default_model
        self.base_url_field.value = preset.base_url
        if preset.api_key_env:
            self.key_hint.value = f"Leave blank to use {preset.api_key_env}"
        self.page.update()

    def _on_threshold_change(self, _event=None) -> None:
        value = float(self.threshold_slider.value or 0.35)
        self.threshold_slider.label = f"{value:.0%}"
        self.threshold_label.value = f"Confidence threshold: {value:.0%}"
        self.page.update()

    def _on_interval_change(self, _event=None) -> None:
        value = float(self.interval_slider.value or 1.0)
        self.interval_slider.label = f"{value:.1f} s"
        self.interval_value.value = f"{value:.1f} s"
        self.page.update()

    # ------------------------------------------------------------- UI updates
    def set_status(self, text: str, kind: str = "idle") -> None:
        palette = {
            "idle": (MUTED, "#F8FAFC"),
            "info": (ACCENT, ACCENT_SOFT),
            "ok": (GREEN, GREEN_SOFT),
            "warn": (RED, RED_SOFT),
            "error": (RED, RED_SOFT),
        }
        color, background = palette.get(kind, palette["idle"])
        self.status.value = text
        self.status.color = color
        self.status_dot.bgcolor = color
        self.status_chip.bgcolor = background
        self.page.update()

    def set_ai_status(self, text: str, kind: str = "info") -> None:
        palette = {
            "info": (ACCENT, ACCENT_SOFT, ft.Icons.MEMORY),
            "ok": (GREEN, GREEN_SOFT, ft.Icons.CHECK_CIRCLE),
            "warn": (AMBER, AMBER_SOFT, ft.Icons.MEMORY),
            "error": (RED, RED_SOFT, ft.Icons.ERROR_OUTLINE),
            "cloud": (ACCENT, ACCENT_SOFT, ft.Icons.CLOUD),
        }
        color, background, icon = palette.get(kind, palette["info"])
        self.ai_text.value = text
        self.ai_text.color = color
        self.ai_chip.bgcolor = background
        self.ai_icon.icon = icon
        self.ai_icon.color = color
        self.page.update()

    def set_checking(self, active: bool) -> None:
        self.check_spinner.visible = active
        self.page.update()

    async def flash_camera(self, cycles: int = 3) -> None:
        """Flash the whole camera container red without repainting the page."""

        try:
            for _ in range(cycles):
                self.camera_flash.visible = True
                self.page.update()
                await asyncio.sleep(0.14)
                self.camera_flash.visible = False
                self.page.update()
                await asyncio.sleep(0.14)
        except Exception:
            pass
        finally:
            try:
                self.camera_flash.visible = False
                self.page.update()
            except Exception:
                pass

    def set_camera_started(self, started: bool) -> None:
        self.camera_placeholder.visible = not started
        self.live_chip.visible = started
        self.start_camera_btn.disabled = False
        self.start_camera_btn.content = "Camera Off" if started else "Start Camera"
        self.start_camera_btn.icon = (
            ft.Icons.VIDEOCAM_OFF if started else ft.Icons.VIDEOCAM
        )
        self.page.update()

    def set_start_detection_enabled(self, enabled: bool) -> None:
        self.start_detection_btn.disabled = not enabled
        self.page.update()

    def set_monitoring(self, active: bool) -> None:
        self.start_detection_btn.disabled = False
        self.start_detection_btn.content = (
            "Stop Detection" if active else "Start Detection"
        )
        self.start_detection_btn.icon = (
            ft.Icons.STOP_CIRCLE if active else ft.Icons.MOTION_PHOTOS_ON
        )
        self.start_detection_btn.bgcolor = RED if active else INK
        self.page.update()

    def show_warning(
        self,
        result: DetectionResult,
        thumbnail: bytes | None = None,
    ) -> None:
        self.warning_title.value = f"{result.label.upper()} DETECTED"
        backend_label = _backend_label(result.backend)
        stamp = time.strftime("%H:%M:%S")
        self.warning_sub.value = (
            f"{result.confidence:.0%} confidence · {backend_label} · {stamp}"
        )
        if thumbnail:
            self.warning_thumb.src = thumbnail
            self.warning_thumb.visible = True
        else:
            self.warning_thumb.visible = False
        self.warning_card.visible = True
        self.page.update()

    def hide_warning(self) -> None:
        self.warning_card.visible = False
        self.page.update()

    # -------------------------------------------------------------- read state
    def get_target(self) -> ObjectSpec | None:
        key = self.object_dropdown.value or DEFAULT_OBJECT_KEY
        if key == CUSTOM_OBJECT_KEY:
            text = (self.custom_field.value or "").strip()
            return make_custom_target(text) if text else None
        return find_object(key)

    def get_interval(self) -> float:
        return float(self.interval_slider.value or 1.0)

    def get_threshold(self) -> float:
        return float(self.threshold_slider.value or 0.35)

    def alert_enabled(self) -> bool:
        return bool(self.alert_switch.value)

    def get_config(self) -> BackendConfig:
        provider = self.provider_dropdown.value or "embedded"
        preset = PROVIDERS[provider]
        return BackendConfig(
            provider=provider,
            model=(self.model_field.value or preset.default_model).strip(),
            base_url=(self.base_url_field.value or preset.base_url).strip(),
            api_key=(self.api_key_field.value or "").strip(),
            threshold=self.get_threshold(),
        )

    def apply_config(self, config: BackendConfig) -> None:
        """Pre-fill the settings panel from environment/default config."""

        if config.provider not in PROVIDERS:
            config.provider = "embedded"
        preset = PROVIDERS[config.provider]
        self.provider_dropdown.value = config.provider
        self.model_field.value = config.model or preset.default_model
        self.base_url_field.value = config.base_url or preset.base_url
        self.api_key_field.value = config.api_key
        self.threshold_slider.value = config.threshold
        self._on_provider_select()
        self._on_threshold_change()
        self._on_object_select()


def _backend_label(key: str) -> str:
    if key in {"embedded", "fake"}:
        return "local model"
    preset = PROVIDERS.get(key)
    return preset.label if preset else key