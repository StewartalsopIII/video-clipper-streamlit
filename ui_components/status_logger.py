# ui_components/status_logger.py
import streamlit as st
import time

def add_log_entry(message: str, level: str = "info"):
    """Appends a new message to the processing log in session state."""
    timestamp = time.strftime("%H:%M:%S")
    if "processing_log" not in st.session_state: # Initialize if called early
        st.session_state.processing_log = []
    st.session_state.processing_log.append({"time": timestamp, "level": level.upper(), "message": message})

def display_log(log_container_placeholder): # Pass the placeholder
    """Displays the processing log from session state."""
    if "processing_log" in st.session_state and st.session_state.processing_log:
        log_html = ""
        for entry in reversed(st.session_state.processing_log): # Newest first
            color = {"INFO": "#555555", "WARNING": "orange", "ERROR": "red", "DEBUG": "#888888"}.get(entry["level"], "inherit")
            # Basic HTML escaping for message
            escaped_message = st.markdown._html.escape(entry['message'])
            log_html += f"<div style='color: {color}; font-family: monospace; font-size: 0.9em;'>{entry['time']} [{entry['level']}] {escaped_message}</div>"
        
        with log_container_placeholder: # Use the passed placeholder
             st.markdown(f"""
             <div style="background-color: #f0f2f6; border: 1px solid #cccccc; padding: 10px; border-radius: 5px; max-height: 300px; overflow-y: auto; overflow-x: hidden; word-wrap: break-word;">
                 {log_html}
             </div>
             """, unsafe_allow_html=True)