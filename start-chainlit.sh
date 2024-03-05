#!/bin/sh -x
source .venv/bin/activate 
chainlit run --headless --port 8000 basic_qa.py 