
"""
Fake Review Detection System
Main Tkinter Application

Supported platforms:
Amazon, Flipkart, Myntra and Meesho.
"""

import threading
import tkinter as tk
from tkinter import messagebox
from urllib.parse import urlsplit

from src.scraper import get_reviews
from src.predict import predict_reviews


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

BG_COLOR = "#f3f5fa"
CARD_COLOR = "#ffffff"
TITLE_COLOR = "#2f3640"
BUTTON_COLOR = "#273c75"
SUCCESS_COLOR = "#218c74"
ERROR_COLOR = "#c0392b"
TEXT_COLOR = "#34495e"
MUTED_COLOR = "#7f8c8d"
BORDER_COLOR = "#dfe4ea"


# --------------------------------------------------
# URL VALIDATION
# --------------------------------------------------

def validate_url(url):
    """Check whether the entered URL has a valid format."""

    try:
        parsed = urlsplit(url.strip())

        return (
            parsed.scheme.lower() in ("http", "https")
            and bool(parsed.hostname)
        )

    except ValueError:
        return False


# --------------------------------------------------
# SUMMARY CARD
# --------------------------------------------------

def create_summary_card(parent, title, value, color):
    """Create a summary card for displaying statistics."""

    card = tk.Frame(
        parent,
        bg=CARD_COLOR,
        highlightbackground=BORDER_COLOR,
        highlightthickness=1
    )

    tk.Label(
        card,
        text=title,
        font=("Arial", 10),
        bg=CARD_COLOR,
        fg=MUTED_COLOR
    ).pack(
        anchor="w",
        padx=15,
        pady=(10, 3)
    )

    value_label = tk.Label(
        card,
        text=str(value),
        font=("Arial", 22, "bold"),
        bg=CARD_COLOR,
        fg=color
    )

    value_label.pack(
        anchor="w",
        padx=15,
        pady=(0, 10)
    )

    return card, value_label


# --------------------------------------------------
# UPDATE SUMMARY CARDS
# --------------------------------------------------

def update_summary_cards(summary):
    """Update summary statistics after prediction."""

    total_value.config(text=str(summary["total"]))

    suspicious_value.config(
        text=str(summary["fake_count"])
    )

    genuine_value.config(
        text=str(summary["total"] - summary["fake_count"])
    )

    percentage_value.config(
        text=f'{summary["percentage_fake"]:.2f}%'
    )


# --------------------------------------------------
# UPDATE ANALYSIS CHART
# --------------------------------------------------

def update_chart(summary):
    """Draw a pie chart showing suspicious and genuine reviews."""

    chart_canvas.delete("all")

    total = summary["total"]
    fake_count = summary["fake_count"]
    genuine_count = total - fake_count

    if total == 0:
        chart_canvas.create_text(
            250,
            65,
            text="No data available",
            font=("Arial", 10),
            fill=MUTED_COLOR
        )
        return

    suspicious_angle = (fake_count / total) * 360
    genuine_angle = (genuine_count / total) * 360

    # Pie chart dimensions.
    x1, y1, x2, y2 = 35, 5, 155, 125

    # Suspicious reviews.
    if fake_count > 0:
        chart_canvas.create_arc(
            x1, y1, x2, y2,
            start=90,
            extent=suspicious_angle,
            fill=ERROR_COLOR,
            outline="white",
            width=2
        )

    # Genuine reviews.
    if genuine_count > 0:
        chart_canvas.create_arc(
            x1, y1, x2, y2,
            start=90 + suspicious_angle,
            extent=genuine_angle,
            fill=SUCCESS_COLOR,
            outline="white",
            width=2
        )

    # Suspicious legend.
    chart_canvas.create_rectangle(
        190, 35, 204, 49,
        fill=ERROR_COLOR,
        outline=ERROR_COLOR
    )

    chart_canvas.create_text(
        215, 42,
        text=(
            f"Suspicious: {fake_count} "
            f"({fake_count / total * 100:.2f}%)"
        ),
        anchor="w",
        font=("Arial", 10),
        fill=TITLE_COLOR
    )

    # Genuine legend.
    chart_canvas.create_rectangle(
        190, 75, 204, 89,
        fill=SUCCESS_COLOR,
        outline=SUCCESS_COLOR
    )

    chart_canvas.create_text(
        215, 82,
        text=(
            f"Genuine: {genuine_count} "
            f"({genuine_count / total * 100:.2f}%)"
        ),
        anchor="w",
        font=("Arial", 10),
        fill=TITLE_COLOR
    )


# --------------------------------------------------
# RESET ANALYSIS CHART
# --------------------------------------------------

def reset_chart():
    """Reset chart when the application is cleared."""

    chart_canvas.delete("all")

    chart_canvas.create_text(
        250,
        65,
        text="Chart will appear after analysis.",
        font=("Arial", 10),
        fill=MUTED_COLOR
    )


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

def display_results(summary, scraped_reviews):
    """Display prediction results and extracted review metadata."""

    output_box.config(state=tk.NORMAL)
    output_box.delete("1.0", tk.END)

    total = summary["total"]
    results = summary["results"]

    if total == 0:
        show_error(
            "No usable reviews were available for prediction."
        )
        return

    # Update summary and chart.
    update_summary_cards(summary)
    update_chart(summary)

    output_box.insert(
        tk.END,
        "INDIVIDUAL REVIEW ANALYSIS\n",
        "section_heading"
    )

    output_box.insert(
        tk.END,
        "Detailed results for the extracted product reviews.\n\n",
        "muted"
    )

    for index, item in enumerate(results):

        prediction = item["prediction"]

        # Get corresponding scraped review metadata.
        metadata = (
            scraped_reviews[index]
            if index < len(scraped_reviews)
            and isinstance(scraped_reviews[index], dict)
            else {}
        )

        review_text = item.get("review", "")

        review_title = (
            metadata.get("review_title") or "Not available"
        )

        rating = metadata.get("rating")

        verified = (
            metadata.get("verified_purchase") or "Unknown"
        )

        category = (
            metadata.get("product_category") or "Unknown"
        )

        if prediction == 1:
            label = "SUSPICIOUS"
            tag = "suspicious"
        else:
            label = "GENUINE"
            tag = "genuine"

        # Review heading.
        output_box.insert(
            tk.END,
            f"REVIEW {index + 1}   ",
            "review_heading"
        )

        output_box.insert(
            tk.END,
            f"[{label}]\n",
            tag
        )

        # Metadata.
        output_box.insert(
            tk.END,
            f"Title: {review_title}\n",
            "metadata"
        )

        output_box.insert(
            tk.END,
            f"Rating: {rating if rating is not None else 'Not available'}\n",
            "metadata"
        )

        output_box.insert(
            tk.END,
            f"Verified Purchase: {verified}\n",
            "metadata"
        )

        output_box.insert(
            tk.END,
            f"Product Category: {category}\n\n",
            "metadata"
        )

        # Review text.
        output_box.insert(
            tk.END,
            "Review Text:\n",
            "review_label"
        )

        output_box.insert(
            tk.END,
            f"{review_text}\n\n",
            "review_text"
        )

        output_box.insert(
            tk.END,
            "-" * 65 + "\n\n",
            "separator"
        )

    # Disclaimer.
    output_box.insert(
        tk.END,
        "IMPORTANT NOTE\n",
        "section_heading"
    )

    output_box.insert(
        tk.END,
        "These predictions are based on the trained machine "
        "learning model. They are experimental and should not "
        "be treated as definitive proof that a review is fake "
        "or genuine. Results may vary across platforms and "
        "depending on the available review information.\n",
        "muted"
    )

    output_box.config(state=tk.DISABLED)

    # Start displaying results from the beginning.
    output_box.yview_moveto(0)

    status_label.config(
        text=f"Analysis completed successfully. {total} reviews analyzed.",
        fg=SUCCESS_COLOR
    )


# --------------------------------------------------
# ERROR HANDLING
# --------------------------------------------------

def show_error(error_message):
    """Display an error message in the results area."""

    output_box.config(state=tk.NORMAL)
    output_box.delete("1.0", tk.END)

    output_box.insert(
        tk.END,
        "ANALYSIS COULD NOT BE COMPLETED\n\n",
        "error_heading"
    )

    output_box.insert(
        tk.END,
        str(error_message),
        "review_text"
    )

    output_box.config(state=tk.DISABLED)


# --------------------------------------------------
# BACKGROUND ANALYSIS
# --------------------------------------------------

def analyze_in_background(url):
    """Scrape product reviews and classify them."""

    try:

        root.after(
            0,
            lambda: status_label.config(
                text="Extracting product reviews...",
                fg=BUTTON_COLOR
            )
        )

        # Step 1: Extract reviews.
        reviews = get_reviews(url)

        if not reviews:
            raise ValueError(
                "No customer reviews could be extracted "
                "from this webpage.\n\n"
                "Possible reasons:\n"
                "1. The website blocks automated requests.\n"
                "2. Reviews are loaded dynamically.\n"
                "3. The webpage uses different HTML structures.\n"
                "4. The product page does not contain accessible reviews.\n\n"
                "Please try another product URL."
            )

        # Step 2: Update status.
        root.after(
            0,
            lambda: status_label.config(
                text=f"{len(reviews)} reviews extracted. Analyzing...",
                fg=BUTTON_COLOR
            )
        )

        # Step 3: Predict reviews.
        summary = predict_reviews(reviews)

        # Step 4: Display results.
        root.after(
            0,
            lambda: display_results(summary, reviews)
        )

    except Exception as error:

        error_message = str(error)

        root.after(
            0,
            lambda message=error_message: show_error(message)
        )

        root.after(
            0,
            lambda: status_label.config(
                text="Analysis failed.",
                fg=ERROR_COLOR
            )
        )

    finally:

        root.after(
            0,
            lambda: analyze_button.config(state=tk.NORMAL)
        )

        root.after(
            0,
            lambda: clear_button.config(state=tk.NORMAL)
        )


# --------------------------------------------------
# START ANALYSIS
# --------------------------------------------------

def analyze_reviews():
    """Validate URL and start background analysis."""

    url = url_entry.get().strip()

    if not url:

        messagebox.showwarning(
            "Missing URL",
            "Please enter a product URL."
        )

        return

    if not validate_url(url):

        messagebox.showerror(
            "Invalid URL",
            "Please enter a valid HTTP or HTTPS URL."
        )

        return

    # Clear previous results.
    output_box.config(state=tk.NORMAL)
    output_box.delete("1.0", tk.END)

    output_box.insert(
        tk.END,
        "Please wait while the product reviews are being analyzed..."
    )

    output_box.config(state=tk.DISABLED)

    # Reset summary cards.
    total_value.config(text="--")
    suspicious_value.config(text="--")
    genuine_value.config(text="--")
    percentage_value.config(text="--")

    # Reset chart.
    reset_chart()

    status_label.config(
        text="Starting analysis...",
        fg=BUTTON_COLOR
    )

    analyze_button.config(state=tk.DISABLED)
    clear_button.config(state=tk.DISABLED)

    # Run analysis in a separate thread.
    thread = threading.Thread(
        target=analyze_in_background,
        args=(url,),
        daemon=True
    )

    thread.start()


# --------------------------------------------------
# CLEAR APPLICATION
# --------------------------------------------------

def clear_application():
    """Clear input and reset the application."""

    url_entry.delete(0, tk.END)

    output_box.config(state=tk.NORMAL)
    output_box.delete("1.0", tk.END)
    output_box.config(state=tk.DISABLED)

    total_value.config(text="--")
    suspicious_value.config(text="--")
    genuine_value.config(text="--")
    percentage_value.config(text="--")

    reset_chart()

    status_label.config(
        text="Ready to analyze product reviews.",
        fg=SUCCESS_COLOR
    )


# --------------------------------------------------
# MAIN WINDOW
# --------------------------------------------------

root = tk.Tk()

root.title("Fake Review Detection System")

root.geometry("1100x800")
root.minsize(850, 650)

root.configure(bg=BG_COLOR)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

header_frame = tk.Frame(
    root,
    bg=BG_COLOR
)

header_frame.pack(
    fill=tk.X,
    pady=(15, 8)
)

title_label = tk.Label(
    header_frame,
    text="Fake Review Detection System",
    font=("Arial", 24, "bold"),
    bg=BG_COLOR,
    fg=TITLE_COLOR
)

title_label.pack()

subtitle_label = tk.Label(
    header_frame,
    text="AI-Based Product Review Analysis using Machine Learning",
    font=("Arial", 10),
    bg=BG_COLOR,
    fg=MUTED_COLOR
)

subtitle_label.pack(pady=(5, 0))


# --------------------------------------------------
# URL INPUT SECTION
# --------------------------------------------------

input_frame = tk.Frame(
    root,
    bg=CARD_COLOR,
    highlightbackground=BORDER_COLOR,
    highlightthickness=1
)

input_frame.pack(
    fill=tk.X,
    padx=30,
    pady=7
)

tk.Label(
    input_frame,
    text="Enter Product URL",
    font=("Arial", 11, "bold"),
    bg=CARD_COLOR,
    fg=TITLE_COLOR
).pack(
    anchor="w",
    padx=15,
    pady=(10, 6)
)

url_entry = tk.Entry(
    input_frame,
    font=("Arial", 11),
    relief=tk.SOLID,
    bd=1,
    bg="#ffffff",
    fg=TITLE_COLOR,
    insertbackground=TITLE_COLOR
)

url_entry.pack(
    fill=tk.X,
    padx=15,
    ipady=7
)

url_entry.bind(
    "<Return>",
    lambda event: analyze_reviews()
)


# --------------------------------------------------
# BUTTONS
# --------------------------------------------------

button_frame = tk.Frame(
    input_frame,
    bg=CARD_COLOR
)

button_frame.pack(pady=10)

analyze_button = tk.Button(
    button_frame,
    text="Analyze Reviews",
    font=("Arial", 10, "bold"),
    bg=BUTTON_COLOR,
    fg="white",
    activebackground=BUTTON_COLOR,
    activeforeground="white",
    padx=23,
    pady=8,
    relief=tk.FLAT,
    cursor="hand2",
    command=analyze_reviews
)

analyze_button.pack(
    side=tk.LEFT,
    padx=6
)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    font=("Arial", 10, "bold"),
    bg="#dcdde1",
    fg=TITLE_COLOR,
    activebackground="#c8cbd1",
    padx=23,
    pady=8,
    relief=tk.FLAT,
    cursor="hand2",
    command=clear_application
)

clear_button.pack(
    side=tk.LEFT,
    padx=6
)


# --------------------------------------------------
# STATUS
# --------------------------------------------------

status_label = tk.Label(
    root,
    text="Ready to analyze product reviews.",
    font=("Arial", 9),
    bg=BG_COLOR,
    fg=SUCCESS_COLOR
)

status_label.pack(pady=(5, 8))


# --------------------------------------------------
# SUMMARY SECTION
# --------------------------------------------------

summary_heading = tk.Label(
    root,
    text="Analysis Summary",
    font=("Arial", 13, "bold"),
    bg=BG_COLOR,
    fg=TITLE_COLOR
)

summary_heading.pack(
    anchor="w",
    padx=30,
    pady=(3, 7)
)

summary_frame = tk.Frame(
    root,
    bg=BG_COLOR
)

summary_frame.pack(
    fill=tk.X,
    padx=30
)

for column in range(4):
    summary_frame.grid_columnconfigure(
        column,
        weight=1
    )

total_card, total_value = create_summary_card(
    summary_frame,
    "Total Reviews",
    "--",
    BUTTON_COLOR
)

suspicious_card, suspicious_value = create_summary_card(
    summary_frame,
    "Suspicious Reviews",
    "--",
    ERROR_COLOR
)

genuine_card, genuine_value = create_summary_card(
    summary_frame,
    "Genuine Reviews",
    "--",
    SUCCESS_COLOR
)

percentage_card, percentage_value = create_summary_card(
    summary_frame,
    "Suspicious Percentage",
    "--",
    ERROR_COLOR
)

total_card.grid(
    row=0, column=0,
    sticky="nsew",
    padx=(0, 5)
)

suspicious_card.grid(
    row=0, column=1,
    sticky="nsew",
    padx=5
)

genuine_card.grid(
    row=0, column=2,
    sticky="nsew",
    padx=5
)

percentage_card.grid(
    row=0, column=3,
    sticky="nsew",
    padx=(5, 0)
)


# --------------------------------------------------
# CHART SECTION
# --------------------------------------------------

chart_heading = tk.Label(
    root,
    text="Review Distribution",
    font=("Arial", 13, "bold"),
    bg=BG_COLOR,
    fg=TITLE_COLOR
)

chart_heading.pack(
    anchor="w",
    padx=30,
    pady=(12, 5)
)

chart_frame = tk.Frame(
    root,
    bg=CARD_COLOR,
    highlightbackground=BORDER_COLOR,
    highlightthickness=1
)

chart_frame.pack(
    fill=tk.X,
    padx=30,
    pady=(0, 5)
)

chart_canvas = tk.Canvas(
    chart_frame,
    width=500,
    height=130,
    bg=CARD_COLOR,
    highlightthickness=0
)

chart_canvas.pack(pady=3)

reset_chart()


# --------------------------------------------------
# RESULTS SECTION
# --------------------------------------------------

result_heading = tk.Label(
    root,
    text="Individual Review Analysis",
    font=("Arial", 13, "bold"),
    bg=BG_COLOR,
    fg=TITLE_COLOR
)

result_heading.pack(
    anchor="w",
    padx=30,
    pady=(10, 5)
)

# Separate frame for results and scrollbar.
# This frame expands to use the remaining window space.

results_frame = tk.Frame(
    root,
    bg=BG_COLOR
)

results_frame.pack(
    fill=tk.BOTH,
    expand=True,
    padx=30,
    pady=(0, 8)
)

output_box = tk.Text(
    results_frame,
    font=("Arial", 10),
    wrap=tk.WORD,
    relief=tk.SOLID,
    bd=1,
    padx=15,
    pady=12,
    bg=CARD_COLOR,
    fg=TEXT_COLOR,
    insertbackground=TITLE_COLOR,
    state=tk.DISABLED
)

output_scrollbar = tk.Scrollbar(
    results_frame,
    orient=tk.VERTICAL,
    command=output_box.yview
)

output_box.configure(
    yscrollcommand=output_scrollbar.set
)

output_box.pack(
    side=tk.LEFT,
    fill=tk.BOTH,
    expand=True
)

output_scrollbar.pack(
    side=tk.RIGHT,
    fill=tk.Y
)


# --------------------------------------------------
# TEXT FORMATTING
# --------------------------------------------------

output_box.tag_configure(
    "section_heading",
    font=("Arial", 12, "bold"),
    foreground=BUTTON_COLOR,
    spacing3=7
)

output_box.tag_configure(
    "review_heading",
    font=("Arial", 10, "bold"),
    foreground=TITLE_COLOR
)

output_box.tag_configure(
    "suspicious",
    font=("Arial", 10, "bold"),
    foreground=ERROR_COLOR
)

output_box.tag_configure(
    "genuine",
    font=("Arial", 10, "bold"),
    foreground=SUCCESS_COLOR
)

output_box.tag_configure(
    "metadata",
    font=("Arial", 9),
    foreground=MUTED_COLOR
)

output_box.tag_configure(
    "review_label",
    font=("Arial", 10, "bold"),
    foreground=TITLE_COLOR
)

output_box.tag_configure(
    "review_text",
    font=("Arial", 10),
    foreground=TEXT_COLOR,
    spacing3=7
)

output_box.tag_configure(
    "muted",
    font=("Arial", 9),
    foreground=MUTED_COLOR,
    spacing3=7
)

output_box.tag_configure(
    "separator",
    foreground=BORDER_COLOR
)

output_box.tag_configure(
    "error_heading",
    font=("Arial", 12, "bold"),
    foreground=ERROR_COLOR
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

footer = tk.Label(
    root,
    text="Project - Soha, Purva, Sanjana, Nikita",
    font=("Arial", 8),
    bg=BG_COLOR,
    fg=MUTED_COLOR
)

footer.pack(pady=(0, 5))


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

root.mainloop()
