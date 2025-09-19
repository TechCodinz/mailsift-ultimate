#!/usr/bin/env python3
"""
MailSift Desktop Application - Standalone Email Extractor
A professional desktop application for email extraction and validation.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import re
import json
import csv
import os
import sys
import threading
import webbrowser
from datetime import datetime
from typing import List, Dict, Any
import requests
import hashlib

class MailSiftDesktop:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MailSift Pro - Email Extractor")
        self.root.geometry("1000x700")
        self.root.configure(bg='#0a0a0a')

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TLabel'
            background='#0a0a0a', foreground='#00ffff')
        self.style.configure('TButton'
            background='#1a1a1a', foreground='#00ffff')
        self.style.configure('TFrame', background='#0a0a0a')

        # Variables
        self.extracted_emails = []
        self.validation_results = {}
        self.license_key = tk.StringVar()

        self.create_widgets()
        self.check_license()

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(header_frame
            text="🚀 MailSift Pro - Professional Email Extractor",
                               font=('Arial', 16, 'bold'))
        title_label.pack()

        subtitle_label = ttk.Label(header_frame
            text="Extract, Validate & Export Emails with AI Intelligence",
                                  font=('Arial', 10))
        subtitle_label.pack()

        # License section
        license_frame = ttk.LabelFrame(main_frame
            text="🔑 License Key", padding=10)
        license_frame.pack(fill=tk.X, pady=(0, 10))

        license_entry = ttk.Entry(license_frame
            textvariable=self.license_key, width=50)
        license_entry.pack(side=tk.LEFT, padx=(0, 10))

        verify_btn = ttk.Button(license_frame
            text="Verify License", command=self.verify_license)
        verify_btn.pack(side=tk.LEFT, padx=(0, 10))

        buy_btn = ttk.Button(license_frame
            text="Buy License", command=self.open_buy_page)
        buy_btn.pack(side=tk.LEFT)

        # Input section
        input_frame = ttk.LabelFrame(main_frame
            text="📝 Input Text or Files", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Text input
        self.text_input = scrolledtext.ScrolledText(input_frame
            height=8, bg='#1a1a1a', fg='#ffffff',
                                                   insertbackground='#00ffff')
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # File input buttons
        file_frame = ttk.Frame(input_frame)
        file_frame.pack(fill=tk.X)

        ttk.Button(file_frame
            text="📁 Open File", command=self.open_file).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="🌐 Extract from URL"
            command=self.extract_from_url).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame
            text="🧹 Clear", command=self.clear_input).pack(side=tk.LEFT)

        # Processing section
        process_frame = ttk.Frame(main_frame)
        process_frame.pack(fill=tk.X, pady=(0, 10))

        self.extract_btn = ttk.Button(process_frame, text="🔍 Extract Emails",
                                     command=self.extract_emails
                                         style='TButton')
        self.extract_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.validate_btn = ttk.Button(process_frame, text="✅ Validate Emails",
                                      command=self.validate_emails
                                          state=tk.DISABLED)
        self.validate_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.progress = ttk.Progressbar(process_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

        # Results section
        results_frame = ttk.LabelFrame(main_frame
            text="📊 Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Results tree
        self.results_tree = ttk.Treeview(results_frame
            columns=('Status', 'Provider', 'Type'), show='tree headings')
        self.results_tree.heading('#0', text='Email Address')
        self.results_tree.heading('Status', text='Status')
        self.results_tree.heading('Provider', text='Provider')
        self.results_tree.heading('Type', text='Type')

        self.results_tree.column('#0', width=300)
        self.results_tree.column('Status', width=100)
        self.results_tree.column('Provider', width=150)
        self.results_tree.column('Type', width=100)

        # Scrollbar for results
        results_scroll = ttk.Scrollbar(results_frame
            orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scroll.set)

        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Export section
        export_frame = ttk.Frame(main_frame)
        export_frame.pack(fill=tk.X)

        ttk.Button(export_frame
            text="💾 Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(export_frame, text="📊 Export Excel"
            command=self.export_excel).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(export_frame, text="📄 Export JSON"
            command=self.export_json).pack(side=tk.LEFT, padx=(0, 10))

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Enter text
            open a file to extract emails")
        status_bar = ttk.Label(main_frame
            textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def check_license(self):
        """Check for saved license key"""
        try:
            config_file = os.path.join(os.path.expanduser('~')
                '.mailsift_config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.license_key.set(config.get('license_key', ''))
                    if self.license_key.get():
                        self.verify_license()
        except Exception:
            pass

    def verify_license(self):
        """Verify license key"""
        key = self.license_key.get().strip()
        if not key:
            messagebox.showwarning("Warning", "Please enter a license key")
            return

        self.status_var.set("Verifying license...")
        self.progress.start()

        def verify():
            try:
                # Simulate license verification (replace with actual API call)
                response
                    requests.post('https://api.mailsift.com/verify-license',
                                       json={'license_key': key}, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('valid'):
                        messagebox.showinfo("Success"
                            "License verified successfully!")
                        self.save_license(key)
                        self.status_var.set("License verified
                            Full features unlocked")
                    else:
                        messagebox.showerror("Error", "Invalid license key")
                        self.status_var.set("Invalid license key")
                else:
                    messagebox.showerror("Error", "Failed to verify license")
                    self.status_var.set("License verification failed")
            except Exception as e:
                messagebox.showerror("Error"
                    f"License verification failed: {str(e)}")
                self.status_var.set("License verification failed")
            finally:
                self.progress.stop()

        threading.Thread(target=verify, daemon=True).start()

    def save_license(self, key):
        """Save license key to config"""
        try:
            config_file = os.path.join(os.path.expanduser('~')
                '.mailsift_config.json')
            config = {'license_key': key
                'last_verified': datetime.now().isoformat()}
            with open(config_file, 'w') as f:
                json.dump(config, f)
        except Exception:
            pass

    def open_buy_page(self):
        """Open buy license page"""
        webbrowser.open('https://mailsift.com/pricing')

    def open_file(self):
        """Open file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select File",
            filetypes=[
                ("All supported", "*.txt;*.csv;*.html;*.pdf;*.docx"),
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("HTML files", "*.html"),
                ("PDF files", "*.pdf"),
                ("Word documents", "*.docx")
            ]
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.text_input.delete(1.0, tk.END)
                    self.text_input.insert(1.0, content)
                    self.status_var.set(f"Loaded file: {os.path.basename(file_p
                        ath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {str(e)}")

    def extract_from_url(self):
        """Extract emails from URL"""
        url = tk.simpledialog.askstring("URL"
            "Enter URL to extract emails from:")
        if url:
            self.status_var.set("Extracting from URL...")
            self.progress.start()

            def extract():
                try:
                    response = requests.get(url, timeout=30)
                    content = response.text
                    self.text_input.delete(1.0, tk.END)
                    self.text_input.insert(1.0, content)
                    self.status_var.set(f"Loaded content from: {url}")
                except Exception as e:
                    messagebox.showerror("Error"
                        f"Failed to load URL: {str(e)}")
                    self.status_var.set("URL extraction failed")
                finally:
                    self.progress.stop()

            threading.Thread(target=extract, daemon=True).start()

    def clear_input(self):
        """Clear input text"""
        self.text_input.delete(1.0, tk.END)
        self.status_var.set("Input cleared")

    def extract_emails(self):
        """Extract emails from input text"""
        text = self.text_input.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning"
                "Please enter some text to extract emails from")
            return

        self.status_var.set("Extracting emails...")
        self.progress.start()

        def extract():
            try:
                # Advanced email regex pattern
                email_pattern
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                emails = re.findall(email_pattern, text)

                # Remove duplicates and sort
                self.extracted_emails = sorted(list(set(emails)))

                # Update UI
                self.root.after(0, self.update_results_display)

            except Exception as e:
                self.root.after(0
                    lambda: messagebox.showerror("Error", f"Extraction failed: {str(e)}"))
                self.root.after(0
                    lambda: self.status_var.set("Extraction failed"))
            finally:
                self.root.after(0, self.progress.stop)

        threading.Thread(target=extract, daemon=True).start()

    def update_results_display(self):
        """Update results display"""
        # Clear existing results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Add extracted emails
        for email in self.extracted_emails:
            provider = email.split('@')[1] if '@' in email else 'Unknown'
            email_type = self.classify_email_type(email)

            self.results_tree.insert('', tk.END, text=email,
                                   values=('Extracted', provider, email_type))

        self.validate_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Found {len(self.extracted_emails)} unique emails"
            )

    def classify_email_type(self, email):
        """Classify email type"""
        domain = email.split('@')[1].lower()

        # Common email providers
        providers = {
            'gmail.com': 'Personal',
            'yahoo.com': 'Personal',
            'hotmail.com': 'Personal',
            'outlook.com': 'Personal',
            'aol.com': 'Personal'
        }

        return providers.get(domain, 'Business')

    def validate_emails(self):
        """Validate extracted emails"""
        if not self.extracted_emails:
            messagebox.showwarning("Warning", "No emails to validate")
            return

        self.status_var.set("Validating emails...")
        self.progress.start()

        def validate():
            try:
                for email in self.extracted_emails:
                    # Basic validation
                    is_valid = self.is_valid_email(email)
                    self.validation_results[email] = {
                        'valid': is_valid,
                        'status': 'Valid' if is_valid else 'Invalid'
                    }

                # Update UI
                self.root.after(0, self.update_validation_results)

            except Exception as e:
                self.root.after(0
                    lambda: messagebox.showerror("Error", f"Validation failed: {str(e)}"))
                self.root.after(0
                    lambda: self.status_var.set("Validation failed"))
            finally:
                self.root.after(0, self.progress.stop)

        threading.Thread(target=validate, daemon=True).start()

    def is_valid_email(self, email):
        """Basic email validation"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def update_validation_results(self):
        """Update validation results in display"""
        for item in self.results_tree.get_children():
            email = self.results_tree.item(item, 'text')
            if email in self.validation_results:
                result = self.validation_results[email]
                self.results_tree.set(item, 'Status', result['status'])

        valid_count = sum(1 for r in self.validation_results.values()
            r['valid'])
        total_count = len(self.validation_results)

        self.status_var.set(f"Validation complete: {valid_count}/{total_count} 
            emails valid")

    def export_csv(self):
        """Export results to CSV"""
        if not self.extracted_emails:
            messagebox.showwarning("Warning", "No emails to export")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Email', 'Status', 'Provider', 'Type'])

                    for email in self.extracted_emails:
                        provider = email.split('@')[1]
                            '@' in email else 'Unknown'
                        email_type = self.classify_email_type(email)
                        status = self.validation_results.get(email
                            {}).get('status', 'Extracted')

                        writer.writerow([email, status, provider, email_type])

                messagebox.showinfo("Success"
                    f"Results exported to {file_path}")
                self.status_var.set(f"Exported {len(self.extracted_emails)} ema
                    ils to CSV")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")

    def export_excel(self):
        """Export results to Excel"""
        messagebox.showinfo("Info"
            "Excel export requires additional setup. Please use CSV export for now.")

    def export_json(self):
        """Export results to JSON"""
        if not self.extracted_emails:
            messagebox.showwarning("Warning", "No emails to export")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )

        if file_path:
            try:
                export_data = {
                    'extracted_at': datetime.now().isoformat(),
                    'total_emails': len(self.extracted_emails),
                    'emails': []
                }

                for email in self.extracted_emails:
                    provider = email.split('@')[1]
                        '@' in email else 'Unknown'
                    email_type = self.classify_email_type(email)
                    status = self.validation_results.get(email
                        {}).get('status', 'Extracted')

                    export_data['emails'].append({
                        'email': email,
                        'status': status,
                        'provider': provider,
                        'type': email_type
                    })

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2)

                messagebox.showinfo("Success"
                    f"Results exported to {file_path}")
                self.status_var.set(f"Exported {len(self.extracted_emails)} ema
                    ils to JSON")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")

    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = MailSiftDesktop()
    app.run()
