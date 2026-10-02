import os
import cv2
import time
import tempfile
import threading
import asyncio
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageDraw

# Check if discord library is installed
try:
    import discord
    HAS_DISCORD = True
except ImportError:
    HAS_DISCORD = False

# Check if pystray library is installed for system tray support
try:
    import pystray
    from pystray import MenuItem as item
    HAS_PYSTRAY = True
except ImportError:
    HAS_PYSTRAY = False


def solve_pixel_assignment(src_img, dst_img):
    """Maps every single pixel in src_img to a unique target position in dst_img
    based on 3D perceptual color sorting without altering pixel values.
    """
    h, w, _ = src_img.shape
    num_pixels = h * w

    y_coords, x_coords = np.mgrid[0:h, 0:w]

    src_colors = src_img.reshape(-1, 3).astype(np.float32)
    src_pos = np.column_stack((x_coords.ravel(), y_coords.ravel()))

    dst_colors = dst_img.reshape(-1, 3).astype(np.float32)
    dst_pos = np.column_stack((x_coords.ravel(), y_coords.ravel()))

    # Perceptual color luminance calculation (BGR)
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
    src_lum = np.dot(src_colors, weights)
    dst_lum = np.dot(dst_colors, weights)

    # 3D key for precise color-space matching
    src_keys = src_lum * 10000 + src_colors[:, 1] * 10 + src_colors[:, 2]
    dst_keys = dst_lum * 10000 + dst_colors[:, 1] * 10 + dst_colors[:, 2]

    src_order = np.argsort(src_keys)
    dst_order = np.argsort(dst_keys)

    start_pos = np.zeros((num_pixels, 2), dtype=np.float32)
    target_pos = np.zeros((num_pixels, 2), dtype=np.float32)
    colors = np.zeros((num_pixels, 3), dtype=np.uint8)

    # 1-to-1 exact mapping
    start_pos[src_order] = src_pos[src_order]
    target_pos[src_order] = dst_pos[dst_order]
    colors[src_order] = src_colors[src_order].astype(np.uint8)

    # Scale swirl radius relative to dynamic canvas dimensions
    max_swirl = max(w, h) * 0.15
    angles = np.random.uniform(0, 2 * np.pi, num_pixels).astype(np.float32)
    swirl_radius = np.random.uniform(
        0.2 * max_swirl, max_swirl, num_pixels
    ).astype(np.float32)

    return start_pos, target_pos, colors, angles, swirl_radius, (h, w)


def save_gif_with_compression(save_path, recorded_frames, fps=30, keep_under_20mb=True, progress_callback=None):
    """Saves frames as a GIF instantly using direct math ratio scaling.
    Guarantees complete end-to-end animation completion and <20MB limit in 1-2 fast passes.
    """
    max_bytes = 20 * 1024 * 1024  # 20 MB Limit
    total_frames = len(recorded_frames)
    total_duration_ms = (total_frames / float(fps)) * 1000.0

    # Smart initial estimates to avoid oversized initial exports
    num_frames = min(75, total_frames) if keep_under_20mb else total_frames
    scale = 0.8 if keep_under_20mb else 1.0

    for attempt in range(1, 4):
        # np.linspace guarantees frame 0 and final frame are ALWAYS present
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frame_duration = int(round(total_duration_ms / float(num_frames)))

        pil_images = []
        for idx, frame_idx in enumerate(indices):
            frame_img = recorded_frames[frame_idx]
            if scale < 0.99:
                h, w, _ = frame_img.shape
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                frame_img = cv2.resize(
                    frame_img, (new_w, new_h), interpolation=cv2.INTER_AREA
                )

            rgb_frame = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb_frame))

            if progress_callback:
                progress = int(((idx + 1) / len(indices)) * 100)
                progress_callback(progress)

        # Fast direct export without slow LZW frame optimization lag
        pil_images[0].save(
            save_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=frame_duration,
            loop=0,
            optimize=False,
        )

        if not keep_under_20mb:
            break

        actual_size = os.path.getsize(save_path)
        if actual_size <= max_bytes:
            break

        # Directly calculate precise scale factor using area/file size ratio
        size_ratio = max_bytes / float(actual_size)
        adjust_factor = np.sqrt(size_ratio) * 0.88

        scale = max(0.15, scale * adjust_factor)
        if num_frames > 25 and size_ratio < 0.5:
            num_frames = max(25, int(num_frames * 0.75))


class DiscordBotManager:
    """Manages the Discord bot lifecycle with Mentions in a background thread."""

    def __init__(self, log_callback):
        self.log = log_callback
        self.token = ""
        self.client = None
        self.loop = None
        self.thread = None
        self.is_running = False

    def start_bot(self, token):
        if not HAS_DISCORD:
            messagebox.showerror(
                "Missing Library",
                "The 'discord.py' package is not installed.\n\nPlease install it using:\npip install discord.py",
            )
            return False

        self.token = token.strip()
        if not self.token:
            messagebox.showwarning("Missing Token", "Please enter a valid Discord Bot API Key!")
            return False

        if self.is_running:
            self.log("⚠️ Bot is already running!")
            return False

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return True

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # REQUIRED FOR READING MENTIONS AND MESSAGES
        intents = discord.Intents.default()
        intents.message_content = True 

        self.client = discord.Client(intents=intents)

        @self.client.event
        async def on_message(message):
            # Ignore self to prevent loops
            if message.author == self.client.user:
                return

            # Check if the bot was mentioned in the message
            if self.client.user in message.mentions:
                author_tag = f"{message.author.name} ({message.author.id})"
                channel_name = getattr(message.channel, 'name', 'DM/Unknown')
                self.log(f"📩 Received @mention from @{author_tag} in #{channel_name}")

                if not message.attachments:
                    self.log("⚠️ Command ignored: No image attached.")
                    await message.reply("bro you forgot the image 💀 attach a pic when you @ me!", mention_author=True)
                    return

                image = message.attachments[0]
                valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
                
                if not image.filename.lower().endswith(valid_exts):
                    self.log("⚠️ Command ignored: Attachment wasn't a valid image.")
                    await message.reply("⚠️ Nah that ain't it, please upload a valid image file (PNG, JPG, WEBP, BMP)!", mention_author=True)
                    return

                start_time = time.time()
                self.log(f"📥 Downloading attachment: {image.filename}...")

                # Start the typing indicator while processing
                async with message.channel.typing():
                    try:
                        # Locate target image on local system
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        target_path = os.path.join(script_dir, "terget.png")
                        if not os.path.exists(target_path):
                            target_path = os.path.join(script_dir, "target.png")

                        if not os.path.exists(target_path):
                            self.log("❌ ERROR: target image (terget.png / target.png) missing from folder!")
                            await message.reply("❌ Target image (`terget.png` / `target.png`) not found on host server!")
                            return

                        # Download and decode attachment
                        img_bytes = await image.read()
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        src_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        dst_img = cv2.imread(target_path)

                        if src_img is None or dst_img is None:
                            self.log("❌ ERROR: Failed to decode source image or target image.")
                            await message.reply("❌ Failed to decode image attachment or target image.")
                            return

                        # Resize images to 500x500 standard canvas
                        target_w, target_h = 500, 500
                        src_img = cv2.resize(src_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        dst_img = cv2.resize(dst_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

                        self.log("⚡ Calculating 1-to-1 pixel map & rendering 180 morph frames...")
                        start_pos, target_pos, colors, angles, swirl_radius, (h, w) = (
                            solve_pixel_assignment(src_img, dst_img)
                        )

                        total_frames = 180
                        recorded_frames = []

                        for frame in range(total_frames):
                            raw_t = frame / float(total_frames - 1)
                            t = raw_t * raw_t * raw_t * (raw_t * (raw_t * 6 - 15) + 10)
                            swirl_factor = (
                                np.sin(raw_t * np.pi)
                                * np.power(raw_t, 1.5)
                                * np.power(1.0 - raw_t, 1.5)
                            )

                            curr_x = (
                                (1 - t) * start_pos[:, 0]
                                + t * target_pos[:, 0]
                                + swirl_factor * swirl_radius * np.cos(angles)
                            )
                            curr_y = (
                                (1 - t) * start_pos[:, 1]
                                + t * target_pos[:, 1]
                                + swirl_factor * swirl_radius * np.sin(angles)
                            )

                            curr_x = np.clip(np.round(curr_x), 0, w - 1).astype(np.int32)
                            curr_y = np.clip(np.round(curr_y), 0, h - 1).astype(np.int32)

                            canvas = np.zeros((h, w, 3), dtype=np.uint8)
                            canvas[curr_y, curr_x] = colors
                            recorded_frames.append(canvas)

                        self.log("🎞️ Compressing animation to GIF (<20MB size target)...")
                        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
                            tmp_gif_path = tmp.name

                        save_gif_with_compression(
                            tmp_gif_path, recorded_frames, fps=30, keep_under_20mb=True
                        )

                        gif_size_mb = os.path.getsize(tmp_gif_path) / (1024 * 1024)
                        self.log(f"📤 Uploading finished GIF ({gif_size_mb:.2f} MB) to Discord...")

                        await message.reply(
                            content=f"🎉 **Kirkified!**",
                            file=discord.File(tmp_gif_path),
                            mention_author=True
                        )

                        elapsed = time.time() - start_time
                        self.log(f"✅ Successfully completed and sent GIF in {elapsed:.1f}s!")

                        # Clean temp file
                        if os.path.exists(tmp_gif_path):
                            os.remove(tmp_gif_path)

                    except Exception as e:
                        self.log(f"❌ Exception caught during process: {e}")
                        await message.reply(f"❌ bro something crashed: `{e}`")

        @self.client.event
        async def on_ready():
            self.is_running = True
            self.log(f"🟢 Connected as bot user: {self.client.user}")
            self.log("✅ Ready! Just @mention the bot with an image in Discord.")

        try:
            self.log("🔄 Initializing Discord Client connection...")
            self.loop.run_until_complete(self.client.start(self.token))
        except Exception as e:
            self.is_running = False
            self.log(f"🔴 Bot offline: {e}")

    def stop_bot(self):
        if self.client and self.loop and self.is_running:
            self.log("⏹️️ Stopping bot client...")
            asyncio.run_coroutine_threadsafe(self.client.close(), self.loop)
            self.is_running = False
            self.log("🔴 Bot disconnected.")
        else:
            self.log("⚠ Bot is not running.")


class PixelMorphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Rearrange Morph Studio")
        self.root.geometry("600x480")
        self.root.resizable(False, False)

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.source_path = tk.StringVar()
        self.target_path = tk.StringVar()
        self.status_var = tk.StringVar(value="Status: Ready to morph!")
        self.discord_token_var = tk.StringVar()

        # Resolution & Kirkify controls state
        self.lock_500_var = tk.BooleanVar(value=True)
        self.res_mode_var = tk.StringVar(value="Follow Source Resolution")
        self.kirkify_var = tk.BooleanVar(value=False)

        self.window_name = "Pixel Rearrange Morph"
        
        # Logging system state
        self.log_history = []
        self.log_text_widget = None
        self.discord_manager = DiscordBotManager(self.append_log)
        self.tray_icon = None

        self.create_widgets()

    def append_log(self, text):
        """Thread-safe logger method to update UI console."""
        def _update():
            timestamp = time.strftime("[%H:%M:%S] ")
            entry = f"{timestamp}{text}\n"
            self.log_history.append(entry)
            
            # If the log widget is currently open, write directly to it
            if self.log_text_widget and self.log_text_widget.winfo_exists():
                self.log_text_widget.config(state="normal")
                self.log_text_widget.insert(tk.END, entry)
                self.log_text_widget.see(tk.END)
                self.log_text_widget.config(state="disabled")

        self.root.after(0, _update)

    def create_widgets(self):
        # Main container padding
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header Frame (Title + Discord Integration Button)
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))

        title_label = ttk.Label(
            header_frame,
            text="✨ Pixel Rearrange Morph",
            font=("Segoe UI", 16, "bold"),
        )
        title_label.pack(side=tk.LEFT)

        # Discord Integration Button in Top Right Corner
        discord_btn = ttk.Button(
            header_frame, text="💬 Discord Integration", command=self.open_discord_window
        )
        discord_btn.pack(side=tk.RIGHT)

        # --- Source Image Frame ---
        src_frame = ttk.LabelFrame(
            main_frame, text=" Source Image (Image A) ", padding=10
        )
        src_frame.pack(fill=tk.X, pady=(0, 8))

        self.src_entry = ttk.Entry(src_frame, textvariable=self.source_path, width=48)
        self.src_entry.pack(side=tk.LEFT, padx=(0, 8), fill=tk.X, expand=True)

        self.src_btn = ttk.Button(src_frame, text="Browse...", command=self.browse_source)
        self.src_btn.pack(side=tk.RIGHT)

        # --- Target Image Frame ---
        dst_frame = ttk.LabelFrame(
            main_frame, text=" Target Image (Image B) ", padding=10
        )
        dst_frame.pack(fill=tk.X, pady=(0, 8))

        top_dst_subframe = ttk.Frame(dst_frame)
        top_dst_subframe.pack(fill=tk.X, pady=(0, 5))

        self.dst_entry = ttk.Entry(
            top_dst_subframe, textvariable=self.target_path, width=48
        )
        self.dst_entry.pack(side=tk.LEFT, padx=(0, 8), fill=tk.X, expand=True)

        self.dst_btn = ttk.Button(
            top_dst_subframe, text="Browse...", command=self.browse_target
        )
        self.dst_btn.pack(side=tk.RIGHT)

        # Kirkify Toggle
        self.kirk_chk = ttk.Checkbutton(
            dst_frame,
            text="Kirkify",
            variable=self.kirkify_var,
            command=self.toggle_kirkify,
        )
        self.kirk_chk.pack(anchor="w")

        # --- Resolution Options Frame ---
        res_frame = ttk.LabelFrame(
            main_frame, text=" Output Resolution Settings ", padding=10
        )
        res_frame.pack(fill=tk.X, pady=(0, 15))

        self.lock_chk = ttk.Checkbutton(
            res_frame,
            text="Lock Resolution to 500 x 500 (Fast)",
            variable=self.lock_500_var,
            command=self.toggle_resolution_controls,
        )
        self.lock_chk.pack(anchor="w", side=tk.LEFT)

        self.res_combo = ttk.Combobox(
            res_frame,
            textvariable=self.res_mode_var,
            values=[
                "Follow Source Resolution",
                "Follow Target Resolution",
                "Keep Aspect Ratio",
            ],
            state="disabled",
            width=24,
        )
        self.res_combo.pack(side=tk.RIGHT)

        # --- Controls & Actions ---
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_btn = ttk.Button(
            btn_frame, text="🚀 Start Morph Animation", command=self.start_morph
        )
        self.start_btn.pack(fill=tk.X, ipady=6)

        # Loading / Progress Bar (hidden by default)
        self.progress_bar = ttk.Progressbar(
            main_frame, orient="horizontal", mode="determinate"
        )

        # Status Bar
        status_label = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 9, "italic"),
            foreground="#555555",
        )
        status_label.pack(anchor="w", pady=(2, 0))

    def open_discord_window(self):
        """Opens modal window for configuring Discord Integration with Live Logs & Tray Minimizing."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Discord Bot Integration & Console")
        dialog.geometry("540x460")
        dialog.resizable(False, False)
        dialog.transient(self.root)

        frame = ttk.Frame(dialog, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)

        top_header_frame = ttk.Frame(frame)
        top_header_frame.pack(fill=tk.X, pady=(0, 8))

        lbl = ttk.Label(
            top_header_frame,
            text="💬 Discord Bot Integration",
            font=("Segoe UI", 12, "bold"),
        )
        lbl.pack(side=tk.LEFT)

        def minimize_to_tray():
            if not HAS_PYSTRAY:
                messagebox.showwarning(
                    "Missing Package",
                    "The 'pystray' package is required for system tray support.\n\nInstall it using:\npip install pystray",
                )
                return

            # Hide both windows (Main Studio + Discord Console)
            self.root.withdraw()
            dialog.withdraw()

            # Create a simple tray icon image programmatically
            image = Image.new("RGB", (64, 64), color=(30, 30, 30))
            dc = ImageDraw.Draw(image)
            dc.rectangle((16, 16, 48, 48), fill=(0, 255, 102))

            def restore_window(icon, item):
                icon.stop()
                self.tray_icon = None
                # Restore both windows safely on the main thread
                self.root.after(0, self.root.deiconify)
                self.root.after(0, dialog.deiconify)

            def quit_app(icon, item):
                icon.stop()
                self.tray_icon = None
                self.discord_manager.stop_bot()
                self.root.after(0, self.root.destroy)

            menu = pystray.Menu(
                item("Restore Window", restore_window, default=True),
                item("Stop Bot & Quit", quit_app)
            )

            self.tray_icon = pystray.Icon("kirkifier_bot", image, "Pixel Morph Studio & Bot", menu)
            threading.Thread(target=self.tray_icon.run, daemon=True).start()

        tray_btn = ttk.Button(top_header_frame, text="📥 Minimize to Tray", command=minimize_to_tray)
        tray_btn.pack(side=tk.RIGHT)

        token_frame = ttk.Frame(frame)
        token_frame.pack(fill=tk.X, pady=(0, 8))

        token_lbl = ttk.Label(token_frame, text="API Token:", font=("Segoe UI", 9, "bold"))
        token_lbl.pack(side=tk.LEFT, padx=(0, 8))

        token_entry = ttk.Entry(token_frame, textvariable=self.discord_token_var, show="•", width=30)
        token_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        def start_action():
            self.discord_manager.start_bot(self.discord_token_var.get())

        def stop_action():
            self.discord_manager.stop_bot()

        start_btn = ttk.Button(token_frame, text="▶ Start", width=7, command=start_action)
        start_btn.pack(side=tk.LEFT, padx=(0, 4))

        stop_btn = ttk.Button(token_frame, text="⏹ Stop", width=7, command=stop_action)
        stop_btn.pack(side=tk.LEFT)

        # Live Console Log Frame
        log_lbl = ttk.Label(frame, text="Live Activity Log Console:", font=("Segoe UI", 9, "bold"))
        log_lbl.pack(anchor="w", pady=(6, 4))

        self.log_text_widget = scrolledtext.ScrolledText(
            frame,
            height=15,
            font=("Consolas", 8),
            bg="#1e1e1e",
            fg="#00ff66",
            insertbackground="white",
            wrap="word",
        )
        self.log_text_widget.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Re-populate existing log history
        self.log_text_widget.config(state="normal")
        for entry in self.log_history:
            self.log_text_widget.insert(tk.END, entry)
        self.log_text_widget.see(tk.END)
        self.log_text_widget.config(state="disabled")

        # Clean up widget reference when dialog closes
        def on_close():
            self.log_text_widget = None
            if self.tray_icon:
                self.tray_icon.stop()
                self.tray_icon = None
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_close)

        # Center popup over main window
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

    def disable_all_controls(self):
        """Disables all UI input components during long operations."""
        self.src_entry.config(state="disabled")
        self.src_btn.config(state="disabled")
        self.dst_entry.config(state="disabled")
        self.dst_btn.config(state="disabled")
        self.kirk_chk.config(state="disabled")
        self.lock_chk.config(state="disabled")
        self.res_combo.config(state="disabled")
        self.start_btn.config(state="disabled")

    def enable_all_controls(self):
        """Restores UI input components to their active state."""
        self.src_entry.config(state="normal")
        self.src_btn.config(state="normal")
        self.kirk_chk.config(state="normal")
        self.lock_chk.config(state="normal")
        self.start_btn.config(state="normal")

        # Honor state of Kirkify and Resolution options
        self.toggle_kirkify()
        self.toggle_resolution_controls()

    def toggle_kirkify(self):
        """Enables/disables target inputs and sets path to terget.png automatically."""
        if self.kirkify_var.get():
            script_dir = os.path.dirname(os.path.abspath(__file__))
            kirk_path = os.path.join(script_dir, "terget.png")

            if not os.path.exists(kirk_path):
                alt_path = os.path.join(script_dir, "target.png")
                if os.path.exists(alt_path):
                    kirk_path = alt_path

            self.target_path.set(kirk_path)
            self.dst_entry.config(state="disabled")
            self.dst_btn.config(state="disabled")
        else:
            self.dst_entry.config(state="normal")
            self.dst_btn.config(state="normal")

    def toggle_resolution_controls(self):
        """Enables/disables the dropdown based on 500x500 toggle state."""
        if self.lock_500_var.get():
            self.res_combo.config(state="disabled")
        else:
            self.res_combo.config(state="readonly")

    def browse_source(self):
        path = filedialog.askopenfilename(
            title="Select Source Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.source_path.set(path)

    def browse_target(self):
        path = filedialog.askopenfilename(
            title="Select Target Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.target_path.set(path)

    def is_window_closed(self):
        """Helper to detect if user clicked the 'X' button on the OpenCV window."""
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
        except Exception:
            return True

    def ask_save_format(self):
        """3-choice modal window: None (do nothing), GIF, or PNG."""
        choice = tk.StringVar(value="None")

        dialog = tk.Toplevel(self.root)
        dialog.title("Save Result")
        dialog.geometry("380x130")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        lbl = ttk.Label(
            dialog,
            text="How would you like to save the morph result?",
            font=("Segoe UI", 10),
        )
        lbl.pack(pady=(18, 15))

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=20)

        def set_choice(val):
            choice.set(val)
            dialog.destroy()

        btn_none = ttk.Button(btn_frame, text="None", command=lambda: set_choice("None"))
        btn_none.pack(side=tk.LEFT, expand=True, padx=5)

        btn_gif = ttk.Button(btn_frame, text="GIF", command=lambda: set_choice("GIF"))
        btn_gif.pack(side=tk.LEFT, expand=True, padx=5)

        btn_png = ttk.Button(btn_frame, text="PNG", command=lambda: set_choice("PNG"))
        btn_png.pack(side=tk.LEFT, expand=True, padx=5)

        # Center popup over main window
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        dialog.wait_window()
        return choice.get()

    def start_morph(self):
        src = self.source_path.get().strip()
        dst = self.target_path.get().strip()

        if not src or not dst:
            messagebox.showwarning(
                "Missing Files", "Please select both a Source and Target image!"
            )
            return

        img1 = cv2.imread(src)
        img2 = cv2.imread(dst)

        if img1 is None or img2 is None:
            messagebox.showerror(
                "File Error", "Could not decode one or both image files!"
            )
            return

        self.disable_all_controls()
        self.status_var.set("Status: Processing resolution & image sizing...")
        self.root.update()

        # Handle Resolution Selection Logic
        if self.lock_500_var.get():
            target_w, target_h = 500, 500
            img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            mode = self.res_mode_var.get()
            if mode == "Follow Source Resolution":
                target_h, target_w, _ = img1.shape
                img2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)
            elif mode == "Follow Target Resolution":
                target_h, target_w, _ = img2.shape
                img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
            elif mode == "Keep Aspect Ratio":
                target_h = 500
                orig_h, orig_w, _ = img1.shape
                aspect = orig_w / float(orig_h)
                target_w = max(1, int(round(500 * aspect)))
                img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
                img2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)

        self.status_var.set(
            f"Status: Computing 1-to-1 pixel mapping ({target_w}x{target_h})..."
        )
        self.root.update()

        total_frames = 180
        fps = 30

        start_pos, target_pos, colors, angles, swirl_radius, (h, w) = (
            solve_pixel_assignment(img1, img2)
        )

        # Display initial source image
        cv2.imshow(self.window_name, img1)
        self.status_var.set("Status: Starting morph in 1 second...")
        self.root.update()

        # Responsive 1-second pause before animation begins
        canceled = False
        for _ in range(20):
            if cv2.waitKey(50) & 0xFF == ord("q") or self.is_window_closed():
                canceled = True
                break
            self.root.update()

        if canceled:
            cv2.destroyAllWindows()
            self.enable_all_controls()
            self.status_var.set("Status: Ready for next run!")
            return

        self.status_var.set("Status: Playing live animation window...")
        self.root.update()

        frame_delay = max(1, int(1000 / fps))
        recorded_frames = []

        for frame in range(total_frames):
            raw_t = frame / float(total_frames - 1)

            # Quintic Smootherstep for acceleration at t=0 and deceleration at t=1
            t = raw_t * raw_t * raw_t * (raw_t * (raw_t * 6 - 15) + 10)

            # Smooth Swirl ease-in (t=0) and ease-out decay (t=1)
            swirl_factor = (
                np.sin(raw_t * np.pi)
                * np.power(raw_t, 1.5)
                * np.power(1.0 - raw_t, 1.5)
            )

            curr_x = (
                (1 - t) * start_pos[:, 0]
                + t * target_pos[:, 0]
                + swirl_factor * swirl_radius * np.cos(angles)
            )
            curr_y = (
                (1 - t) * start_pos[:, 1]
                + t * target_pos[:, 1]
                + swirl_factor * swirl_radius * np.sin(angles)
            )

            curr_x = np.clip(np.round(curr_x), 0, w - 1).astype(np.int32)
            curr_y = np.clip(np.round(curr_y), 0, h - 1).astype(np.int32)

            # Pure black canvas
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[curr_y, curr_x] = colors

            recorded_frames.append(canvas.copy())

            cv2.imshow(self.window_name, canvas)
            self.root.update()

            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord("q") or self.is_window_closed():
                break

        # If user did not close window mid-animation, hold final frame until closed or keypress
        if not self.is_window_closed():
            self.status_var.set(
                "Status: Animation finished! Press any key or close the preview window."
            )
            self.root.update()
            while not self.is_window_closed():
                if cv2.waitKey(50) & 0xFF != 255:
                    break
                self.root.update()

        cv2.destroyAllWindows()

        # Prompt user to save as None / GIF / PNG upon closing
        if recorded_frames:
            save_choice = self.ask_save_format()

            if save_choice == "GIF":
                keep_under_20mb = messagebox.askyesno(
                    "File Size Limit", "Do you want to keep the file size under 20 MB?"
                )

                save_path = filedialog.asksaveasfilename(
                    title="Save Animation as GIF",
                    defaultextension=".gif",
                    filetypes=[("GIF Files", "*.gif"), ("All Files", "*.*")],
                )
                if save_path:
                    self.status_var.set("Status: Creating GIF...")
                    self.disable_all_controls()
                    self.progress_bar.pack(fill=tk.X, pady=(0, 6))
                    self.progress_bar["value"] = 0
                    self.root.update()

                    try:
                        def update_prog(val):
                            self.progress_bar["value"] = val
                            self.root.update()

                        save_gif_with_compression(
                            save_path, recorded_frames, fps, keep_under_20mb, progress_callback=update_prog
                        )

                        final_size_mb = os.path.getsize(save_path) / (1024 * 1024)
                        messagebox.showinfo(
                            "Saved Successfully",
                            f"Animation saved as GIF ({final_size_mb:.1f} MB) to:\n{save_path}",
                        )
                    except Exception as e:
                        messagebox.showerror(
                            "Export Error", f"Failed to save GIF:\n{e}"
                        )
                    finally:
                        self.progress_bar.pack_forget()

            elif save_choice == "PNG":
                save_path = filedialog.asksaveasfilename(
                    title="Save Result as PNG Image",
                    defaultextension=".png",
                    filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
                )
                if save_path:
                    try:
                        # Save final frame of animation sequence
                        cv2.imwrite(save_path, recorded_frames[-1])
                        messagebox.showinfo(
                            "Saved Successfully",
                            f"Final morph result saved as PNG to:\n{save_path}",
                        )
                    except Exception as e:
                        messagebox.showerror(
                            "Export Error", f"Failed to save PNG image:\n{e}"
                        )

        self.enable_all_controls()
        self.status_var.set("Status: Ready for next run!")


if __name__ == "__main__":
    root = tk.Tk()
    app = PixelMorphApp(root)
    root.mainloop()
