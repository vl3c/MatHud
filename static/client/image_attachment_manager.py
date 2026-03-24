"""Image attachment manager for the AI interface.

Manages the image attachment lifecycle: file picking, reading, preview
area updates, limit enforcement, and the full-size image modal.

Extracted from ``AIInterface`` to reduce god-class complexity while
preserving the identical public behaviour.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from browser import document, html, window

from constants import IMAGE_SIZE_WARNING_BYTES, MAX_ATTACHED_IMAGES


class ImageAttachmentManager:
    """Manages image attachment state, DOM previews, and the image modal.

    Attributes:
        _images: Data URLs of currently attached images.
        _on_system_message: Optional callback to display system messages in chat.
    """

    def __init__(self, on_system_message: Optional[Callable[[str], None]] = None) -> None:
        self._images: list[str] = []
        self._on_system_message = on_system_message

    # ── Public API ──────────────────────────────────────────────

    @property
    def images(self) -> list[str]:
        """Return the list of currently attached image data URLs."""
        return self._images

    def initialize(self) -> None:
        """Initialize image attachment functionality.

        Binds event handlers for the attach button, file input, and modal.
        Should be called after the DOM is ready.
        """
        try:
            # Bind attach button click
            if "attach-button" in document:
                document["attach-button"].bind("click", self._on_attach_button_click)

            # Bind file input change
            if "image-attach-input" in document:
                document["image-attach-input"].bind("change", self._on_files_selected)

            # Bind modal close handlers
            if "image-modal" in document:
                modal = document["image-modal"]
                modal.bind("click", self._on_modal_backdrop_click)

            close_btn = document.select_one(".image-modal-close")
            if close_btn:
                close_btn.bind("click", self._close_modal)
        except Exception as e:
            print(f"Error initializing image attachment: {e}")

    def trigger_file_picker(self) -> None:
        """Programmatically trigger the file picker for image attachment.

        This is called by the /attach slash command.
        """
        self._on_attach_button_click(None)

    def clear(self) -> None:
        """Clear all attached images."""
        self._images = []
        self._update_preview_area()

    def show_modal(self, data_url: str) -> None:
        """Display an image in full-size modal."""
        try:
            modal = document["image-modal"]
            modal_img = document["image-modal-img"]
            modal_img.src = data_url
            modal.style.display = "flex"
        except Exception as e:
            print(f"Error showing image modal: {e}")

    # ── Internal handlers ───────────────────────────────────────

    def _on_attach_button_click(self, event: Any) -> None:
        """Handle attach button click - trigger file picker."""
        try:
            if "image-attach-input" in document:
                document["image-attach-input"].click()
        except Exception as e:
            print(f"Error triggering file picker: {e}")

    def _on_files_selected(self, event: Any) -> None:
        """Handle file input change - read selected files as data URLs."""
        try:
            file_input = event.target
            files = file_input.files

            if not files or files.length == 0:
                return

            # Check if we've hit the limit
            current_count = len(self._images)
            remaining = MAX_ATTACHED_IMAGES - current_count

            if remaining <= 0:
                if self._on_system_message:
                    self._on_system_message(
                        f"Maximum of {MAX_ATTACHED_IMAGES} images per message. Remove some to add more."
                    )
                file_input.value = ""
                return

            files_to_process = min(files.length, remaining)
            if files.length > remaining:
                if self._on_system_message:
                    self._on_system_message(
                        f"Only attaching {remaining} of {files.length} images (limit: {MAX_ATTACHED_IMAGES})."
                    )

            for i in range(files_to_process):
                file = files[i]
                self._read_and_attach_image(file)

            # Clear the input so the same file can be selected again
            file_input.value = ""
        except Exception as e:
            print(f"Error handling file selection: {e}")

    def _read_and_attach_image(self, file: Any) -> None:
        """Read an image file and add it to the attached images list."""
        try:
            # Check file size
            if hasattr(file, "size") and file.size > IMAGE_SIZE_WARNING_BYTES:
                size_mb = file.size / (1024 * 1024)
                if self._on_system_message:
                    self._on_system_message(
                        f"Warning: Image '{file.name}' is {size_mb:.1f}MB. Large images may slow down processing."
                    )

            # Create FileReader to convert to data URL
            reader = window.FileReader.new()

            def on_load(event: Any) -> None:
                try:
                    data_url = reader.result
                    if isinstance(data_url, str) and data_url.startswith("data:image"):
                        self._images.append(data_url)
                        self._update_preview_area()
                except Exception as e:
                    print(f"Error processing image: {e}")

            reader.onload = on_load
            reader.readAsDataURL(file)
        except Exception as e:
            print(f"Error reading image file: {e}")

    def _update_preview_area(self) -> None:
        """Update the image preview area to reflect current attached images."""
        try:
            preview_area = document["image-preview-area"]

            # Clear existing previews
            preview_area.clear()

            if not self._images:
                preview_area.style.display = "none"
                return

            preview_area.style.display = "flex"

            for idx, data_url in enumerate(self._images):
                # Create preview item container
                item = html.DIV(Class="image-preview-item")

                # Create thumbnail image
                img = html.IMG(src=data_url)
                item <= img

                # Create remove button
                remove_btn = html.BUTTON("\u00d7", Class="remove-btn")
                remove_btn.attrs["title"] = "Remove image"

                # Bind remove handler with closure for index
                def make_remove_handler(index: int) -> Any:
                    def handler(event: Any) -> None:
                        event.stopPropagation()
                        self._remove_image(index)

                    return handler

                remove_btn.bind("click", make_remove_handler(idx))
                item <= remove_btn

                preview_area <= item
        except Exception as e:
            print(f"Error updating preview area: {e}")

    def _remove_image(self, index: int) -> None:
        """Remove an attached image by index."""
        try:
            if 0 <= index < len(self._images):
                self._images.pop(index)
                self._update_preview_area()
        except Exception as e:
            print(f"Error removing attached image: {e}")

    def _close_modal(self, event: Any = None) -> None:
        """Close the image modal."""
        try:
            modal = document["image-modal"]
            modal.style.display = "none"
        except Exception as e:
            print(f"Error closing image modal: {e}")

    def _on_modal_backdrop_click(self, event: Any) -> None:
        """Close modal when clicking outside the image."""
        try:
            if event.target.id == "image-modal":
                self._close_modal()
        except Exception as e:
            print(f"Error handling modal click: {e}")
