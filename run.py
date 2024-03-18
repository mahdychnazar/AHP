import streamlit as st

import streamlit.web.cli as stcli
import sys


if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        "main.py",
        "--global.developmentMode=false",
    ]

    sys.exit(stcli.main())