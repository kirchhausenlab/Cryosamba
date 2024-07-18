import os

import streamlit as st


def list_directories_in_directory(directory):
    try:
        with os.scandir(directory) as entries:
            dirs = [entry.name for entry in entries if entry.is_dir()]
        return dirs
    except PermissionError:
        st.error("Permission denied to access this directory.")
        return []


def get_dir():
    # Set default directory to the current folder
    if "current_directory" not in st.session_state:
        st.session_state.current_directory = os.getcwd()

    st.title("Directory Navigator")

    # Display the current directory
    st.write(f"Current Directory: {st.session_state.current_directory}")
    selected_subdir = st.selectbox(
            "Subdirectories:",
            [""] + list_directories_in_directory(st.session_state.current_directory),
        )
    # Buttons to navigate up and down the directory levels
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Go Up"):
            st.session_state.current_directory = os.path.dirname(
                st.session_state.current_directory
            )

    with col2:
       
        
        if st.button("Go Down") and selected_subdir:
            st.session_state.current_directory = os.path.join(
                st.session_state.current_directory, selected_subdir
            )

    if st.button("Select Path to be displayed below by hitting submit"):
        st.session_state.current_directory = os.path.join(
            st.session_state.current_directory, selected_subdir
        )

    # Display the selected directory
    st.write("Selected Directory:")
    st.write(st.session_state.current_directory)
